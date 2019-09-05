import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules.transformer import TransformerConcatEncoder, TransformerConcatDecoder
from models import *
from utils import count_parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, time_step, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False):
        """
        Construct a basic Transfomer model for multimodal tasks.
        
        :param input_dims: The input dimensions of the various modalities.
        :param hidden_size: The hidden dimensions of the fc layer.
        :param embed_dim: The dimensions of the embedding layer.
        :param output_dim: The dimensions of the output (128 in MuiscNet).
        :param num_heads: The number of heads to use in the multi-headed attention. 
        :param attn_dropout: The dropout following self-attention sm((QK)^T/d)V.
        :param relu_droput: The dropout for ReLU in residual block.
        :param res_dropout: The dropout of each residual block.
        :param out_dropout: The dropout of output layer.
        :param layers: The number of transformer blocks.
        :param attn_mask: A boolean indicating whether to use attention mask (for transformer decoder).
        """
        super(TransformerModel, self).__init__()
        [self.orig_d_a, self.orig_d_b] = input_dims
        assert self.orig_d_a == self.orig_d_b
        self.d_x = 1024
        final_out = embed_dim
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        
        # Transformer networks
        self.trans = self.get_network()
        print("Encoder Model size: {0}".format(count_parameters(self.trans)))
        # Projection layers
        self.fc = nn.Linear(self.orig_d_a + self.orig_d_b, self.d_x)
        self.proj = nn.Linear(self.d_x, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(out_dropout)
    def get_network(self):
        
        return TransformerConcatEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)
            
    def forward(self, x):
        """
        x should have dimension [batch_size, seq_len, n_features] (i.e., N, L, C).
        """
        time_step, batch_size, n_features = x.shape
        x = self.fc(x)
        x = self.proj(x)
        h_x = self.trans(x)
        h_concat = torch.cat([h_x[-1]], dim=-1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))
        # No sigmoid because we use BCEwithlogitis which contains sigmoid layer and more stable
        return output

class TransformerGenerationModel(nn.Module):
    def __init__(self, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False, src_mask=False, tgt_mask=False):
        super(TransformerGenerationModel, self).__init__()
        [orig_d_a, orig_d_b] = input_dims
        self.orig_d_x = orig_d_a + orig_d_b
        self.d_x = 1024
        final_out = embed_dim
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        
        # Transformer networks
        self.trans_encoder = self.get_encoder_network()
        self.trans_decoder = self.get_decoder_network()

        print("Encoder Model size: {0}".format(count_parameters(self.trans_encoder)))
        print("Decoder Model size: {0}".format(count_parameters(self.trans_decoder)))
        
        # Projection layers
        self.fc = nn.Linear(self.orig_d_x, self.d_x)
        self.proj_enc = nn.Linear(self.d_x, self.embed_dim)
        self.proj_dec = nn.Linear(self.orig_d_x, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
   
        self.out_fc3 = nn.Linear(output_dim, 1000)
        
        self.out_dropout = nn.Dropout(out_dropout)

    def get_encoder_network(self):
        
        return TransformerConcatEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)

    def get_decoder_network(self): 
        return TransformerConcatDecoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, src_attn_dropout=self.attn_dropout, 
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, tgt_attn_dropout=self.attn_dropout)
            
    def forward(self, x, y=None, max_len=None):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).  
        """

        time_step, batch_size, n_features = x.shape
        # encoder
        x = self.fc(x)
        x = self.proj_enc(x)
        h_x = self.trans_encoder(x)
        
        # decoder
        if y is not None:
            seq_len, batch_size, n_features2 = y.shape
            y = y[:-1, :, :]                               # truncate last target 
            sos = torch.zeros(1, batch_size, n_features2).cuda()
            y = torch.cat([sos, y], dim=0)    # add <sos> to front 
            y = self.proj_dec(y)
            out = self.trans_decoder(input=y, enc=h_x)
            out_concat = torch.cat([out], dim=-1)
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))
        elif max_len is not None:
            dec_x = torch.zeros(1, batch_size, n_features).cuda()
            dec_x = self.proj_dec(dec_x)
            dec_x = self.trans_decoder(input=dec_x, enc=h_x)
            y = dec_x
            for i in range(max_len - 1):
                dec_x = self.trans_decoder(input=y, enc=h_x)
                y = torch.cat([y, dec_x[-1].unsqueeze(0)], dim=0)
            out_concat = torch.cat([y], dim=-1)
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))

        else:
            print("Only one of y and max_len should be input.")
            assert False
        output = F.relu(self.out_fc3(output[-1])) # use last time step to generate a label for entire sequence
        return output
