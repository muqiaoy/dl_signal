import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules.transformer import TransformerEncoder, TransformerDecoder
from models import *
from utils import count_parameters
from conv import Conv1d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, ntokens, time_step, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, embed_dropout, layers, attn_mask=False, crossmodal=False):
        """
        Construct a basic Transfomer model for multimodal tasks.
        
        :param ntokens: The number of unique tokens in text modality.
        :param input_dims: The input dimensions of the various (in this case, 3) modalities.
        :param num_heads: The number of heads to use in the multi-headed attention. 
        :param attn_dropout: The dropout following self-attention sm((QK)^T/d)V.
        :param relu_droput: The dropout for ReLU in residual block.
        :param res_dropout: The dropout of each residual block.
        :param layers: The number of transformer blocks.
        :param attn_mask: A boolean indicating whether to use attention mask (for transformer decoder).
        :param crossmodal: Use Crossmodal Transformer or Not
        """
        super(TransformerModel, self).__init__()
        self.cnn = ComplexSequential(
            ComplexConv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            ComplexBatchNorm1d(16),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            ComplexBatchNorm1d(128),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),
            ComplexFlatten(),
            )
        [self.orig_d_a, self.orig_d_b] = input_dims
        assert self.orig_d_a == self.orig_d_b
        channels = ((((((((((self.orig_d_a -6)//2+1 -2)//2+1 -3)//2+1 -2)//2+1 
            -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        self.d_a, self.d_b = 128*channels, 128*channels
        self.ntokens = ntokens
        final_out = embed_dim * 2
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        self.crossmodal = crossmodal
        
        # Transformer networks
        self.trans = self.get_network()
        print("Encoder Model size: {0}".format(count_parameters(self.trans)))
        # Projection layers
        self.proj = ComplexLinear(self.d_a, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(out_dropout)
    def get_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, embed_dropout=self.embed_dropout, attn_mask=self.attn_mask, crossmodal=self.crossmodal)
            
    def forward(self, x):
        """
        x should have dimension [batch_size, seq_len, n_features] (i.e., N, L, C).
        """
        time_step, batch_size, n_features = x.shape

        # even_indices = torch.tensor([i for i in range(n_features) if i % 2 == 0]).to(device=device)
        # odd_indices = torch.tensor([i for i in range(n_features) if i % 2 == 1]).to(device=device)
        # input_a = torch.index_select(x, 2, even_indices).view(-1, 1, n_features//2) # (bs, input_size/2) 
        # input_b = torch.index_select(x, 2, odd_indices).view(-1, 1, n_features//2) # (bs, input_size/2) 
        input_a = x[:, :, :n_features//2].view(-1, 1, n_features//2)
        input_b = x[:, :, n_features//2:].view(-1, 1, n_features//2)

        input_a, input_b = self.cnn(input_a, input_b)
        input_a = input_a.reshape(time_step, batch_size, self.d_a)
        input_b = input_b.reshape(time_step, batch_size, self.d_b)
        input_a, input_b = self.proj(input_a, input_b)
        # Pass the input through individual transformers
        h_as, h_bs = self.trans(input_a, input_b)
        h_concat = torch.cat([h_as, h_bs], dim=-1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))
        # No sigmoid because we use BCEwithlogitis which contains sigmoid layer and more stable
        return output

class TransformerGenerationModel(nn.Module):
    def __init__(self, ntokens, input_dims, hidden_size, embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout, layers, horizons, attn_mask=False, src_mask=False, tgt_mask=False, crossmodal=False):
        """
        Construct a basic Transfomer model for multimodal tasks.
        
        :param ntokens: The number of unique tokens in text modality.
        :param input_dims: The input dimensions of the various (in this case, 3) modalities.
        :param num_heads: The number of heads to use in the multi-headed attention. 
        :param attn_dropout: The dropout following self-attention sm((QK)^T/d)V.
        :param relu_droput: The dropout for ReLU in residual block.
        :param res_dropout: The dropout of each residual block.
        :param layers: The number of transformer blocks.
        :param attn_mask: A boolean indicating whether to use attention mask (for transformer decoder).
        :param crossmodal: Use Crossmodal Transformer or Not

        l = a, a = b 
        """
        super(TransformerGenerationModel, self).__init__()
        self.cnn = ComplexSequential(
            ComplexConv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            ComplexBatchNorm1d(16),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            ComplexBatchNorm1d(128),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),
            ComplexFlatten(),
            )
        [self.orig_d_a, self.orig_d_b] = input_dims
        assert self.orig_d_a == self.orig_d_b
        channels = ((((((((((self.orig_d_a -6)//2+1 -2)//2+1 -3)//2+1 -2)//2+1 
            -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        self.d_a, self.d_b = 128*channels, 128*channels
        self.ntokens = ntokens
        final_out = embed_dim * 2
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        self.crossmodal = crossmodal
        # self.src_mask = src_mask  # for decoder
        # self.tgt_mask = tgt_mask  # for decoder
        
        # Transformer networks
        self.trans_encoder = self.get_encoder_network()
        self.trans_decoder = self.get_decoder_network()

        print("Encoder Model size: {0}".format(count_parameters(self.trans_encoder)))
        print("Decoder Model size: {0}".format(count_parameters(self.trans_decoder)))
        
        # Projection layers
        self.proj_enc = ComplexLinear(self.d_a, self.embed_dim)
        self.proj_dec = ComplexLinear(self.orig_d_a, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, self.orig_d_a + self.orig_d_b)
        
        self.out_dropout = nn.Dropout(0.5)

    def get_encoder_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask, crossmodal=self.crossmodal)

    def get_decoder_network(self): 
        return TransformerDecoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, src_attn_dropout=self.attn_dropout, 
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, tgt_attn_dropout=self.attn_dropout, crossmodal=self.crossmodal)
            
    def forward(self, x, y):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).  
        """

        time_step, batch_size, n_features = x.shape
        # even_indices = torch.tensor([i for i in range(n_features) if i % 2 == 0]).to(device=device)
        # odd_indices = torch.tensor([i for i in range(n_features) if i % 2 == 1]).to(device=device)
        # input_a = torch.index_select(x, 2, even_indices).view(-1, 1, n_features//2) # (bs, input_size/2) 
        # input_b = torch.index_select(x, 2, odd_indices).view(-1, 1, n_features//2) # (bs, input_size/2) 
        input_a = x[:, :, :n_features//2].view(-1, 1, n_features//2)
        input_b = x[:, :, n_features//2:].view(-1, 1, n_features//2)

        input_a, input_b = self.cnn(input_a, input_b)
        input_a = input_a.reshape(-1, batch_size, self.d_a)
        input_b = input_b.reshape(-1, batch_size, self.d_b)
        input_a, input_b = self.proj_enc(input_a, input_b)
        # Pass the input through individual transformers
        h_as, h_bs = self.trans_encoder(input_a, input_b)


        # Some interesting behavior here;
        # First the larger the model size, the larger the range
        # Second the data pattern will be (+, -, +, -, ...) if time_step = 5
        # Should choose time_step = even number

        # Fuse the hidden vectors from each modality, and pass through one multimodal transformer
        # last_h_ls = h_ls[-1]
        # last_h_as = h_as[-1]
        # h_concat = torch.cat([last_h_ls, last_h_as], dim=-1)
        # h_concat = self.proj(h_concat)

        # concat all time steps
        # h_ls = torch.transpose(h_ls, 0, 1)
        # h_as = torch.transpose(h_as, 0, 1) 
        # h_ls = h_ls.reshape(batch_size, -1)
        # h_as = h_as.reshape(batch_size, -1) 
        # h_concat = torch.cat([h_ls, h_as], dim=-1)

         # concat last time steps of all horizons 
        # h_ls_as_each_catted = [torch.cat([h_ls_as[i][0][-1], h_ls_as[i][1][-1]], dim=-1) for i in range(self.horizons)]
        # h_concat = torch.cat(h_ls_as_each_catted, dim=-1)

        # output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))

        # print("h_ls_as[0][0].shape") # (list dim = horizons, tuple dim = 2, Tensor(10 (TS), 256 (BS), 160 (a_dim/b_dim)))
        # print(h_ls_as[0][0].shape) 
        # print("h_ls_as_each_catted[0].shape") # (horizons, Tensor(BS, a_dim + b_dim))
        # print(h_ls_as_each_catted[0].shape)
        # print("h_concat.shape")  # (BS, (a_dim + b_dim) * horizons) 
        # print(h_concat.shape) 
        # print("output.shape") $ (BS, output_dim)
        # print(output.shape)
        # exit()

        # DECODER: ASSUME self.horizons = 1
        # print(y.shape)
        # print(self.orig_d_l)

        batch_size, seq_len, n_features2 = y.shape 
        n_features = n_features2 // 2

        y_a = y[:, :-1, :self.orig_d_a]                               # truncate last target 
        y_b = y[:, :-1, self.orig_d_a: self.orig_d_a + self.orig_d_b] # truncate last target 

        sos_a = torch.zeros(batch_size, 1, n_features).cuda()
        sos_b = torch.zeros(batch_size, 1, n_features).cuda()
        y_a = torch.cat([sos_a, y_a], dim=1)    # add <sos> to front 
        y_b = torch.cat([sos_b, y_b], dim=1)    # add <sos> to front 

        y_a, y_b = self.proj_dec(y_a, y_b)
        out_as, out_bs = self.trans_decoder(input_A=y_a, input_B=y_b, enc_A=h_as, enc_B=h_bs)

        # out_ls = out_ls[:-1]  # no need to slice if we <sos> to front and truncate last 
        # out_as = out_as[:-1]

        out_concat = torch.cat([out_as, out_bs], dim=-1)
        
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))


        return output # (TS, BS, feature_dim)  
        # return output, h_ls_as   # (list dim = horizons, tuple dim = 2, Tensor(10 (TS), 256 (BS), 160 (a_dim/b_dim)))
