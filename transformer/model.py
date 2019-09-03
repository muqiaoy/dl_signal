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
        self.conv = ComplexSequential(
            ComplexConv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1),
            ComplexBatchNorm1d(16),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
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
        channels = ((((((((((self.orig_d_a -6)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 
            -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        self.d_a, self.d_b = 128*channels, 128*channels
        final_out = embed_dim * 2
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
        self.proj = ComplexLinear(self.d_a, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(out_dropout)
    def get_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)
            
    def forward(self, x):
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

        input_a, input_b = self.conv(input_a, input_b)
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
    def __init__(self, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False, src_mask=False, tgt_mask=False):
        super(TransformerGenerationModel, self).__init__()
        self.conv = ComplexSequential(
            ComplexConv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1),
            ComplexBatchNorm1d(16),
            ComplexReLU(),
            ComplexMaxPool1d(2, stride=2),

            ComplexConv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
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
        channels = ((((((((((self.orig_d_a -6)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 
            -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        self.d_a, self.d_b = 128*channels, 128*channels
        final_out = embed_dim * 2
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
        self.proj_enc = ComplexLinear(self.d_a, self.embed_dim)
        self.proj_dec = ComplexLinear(self.orig_d_a, self.embed_dim)
        
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(out_dropout)

    def get_encoder_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)

    def get_decoder_network(self): 
        return TransformerDecoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, src_attn_dropout=self.attn_dropout, 
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, tgt_attn_dropout=self.attn_dropout)
            
    def forward(self, x, y=None, max_len=None, start=None):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).  
        """
        time_step, batch_size, n_features = x.shape
        input_a = x[:, :, :n_features//2].view(-1, 1, n_features//2)
        input_b = x[:, :, n_features//2:].view(-1, 1, n_features//2)
        print("input_a, input_b", input_a.mean().item(), input_b.mean().item())
        input_a, input_b = self.conv(input_a, input_b)
        input_a = input_a.reshape(-1, batch_size, self.d_a)
        input_b = input_b.reshape(-1, batch_size, self.d_b)
        print("input_a, input_b", input_a.mean().item(), input_b.mean().item())
        input_a, input_b = self.proj_enc(input_a, input_b)
        print("input_a, input_b", input_a.mean().item(), input_b.mean().item())
        # Pass the input through individual transformers
        h_as, h_bs = self.trans_encoder(input_a, input_b)
        print("h_as, h_bs", h_as.mean().item(), h_bs.mean().item())

        if y is not None:
            seq_len, batch_size, n_features2 = y.shape 
            n_features = n_features2 // 2

            y_a = y[:-1, :, :self.orig_d_a]                               # truncate last target 
            y_b = y[:-1, :, self.orig_d_a: self.orig_d_a + self.orig_d_b] # truncate last target 

            sos_a = torch.zeros(1, batch_size, n_features).cuda()
            sos_b = torch.zeros(1, batch_size, n_features).cuda()
            y_a = torch.cat([sos_a, y_a], dim=0)    # add <sos> to front 
            y_b = torch.cat([sos_b, y_b], dim=0)    # add <sos> to front 

            y_a, y_b = self.proj_dec(y_a, y_b)
            print("y_a, y_b", y_a.mean().item(), y_b.mean().item())
            out_as, out_bs = self.trans_decoder(input_A=y_a, input_B=y_b, enc_A=h_as, enc_B=h_bs)
            #print(out_as.mean().item(), out_bs.mean().item())

            # out_ls = out_ls[:-1]  # no need to slice if we <sos> to front and truncate last 
            # out_as = out_as[:-1]
            print("out_as, out_bs", out_as.mean().item(), out_bs.mean().item())
            out_concat = torch.cat([out_as, out_bs], dim=-1)
            print("out_concat", out_concat.mean().item())
            
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))

        elif max_len is not None:
            if start is None:
                #dec_a = torch.rand(1, batch_size, n_features//2).cuda()
                #dec_b = torch.rand(1, batch_size, n_features//2).cuda()
                #dec_a = torch.ones(1, batch_size, n_features//2).cuda()
                #dec_b = torch.ones(1, batch_size, n_features//2).cuda()
                dec_a = torch.zeros(1, batch_size, n_features//2).cuda()
                dec_b = torch.zeros(1, batch_size, n_features//2).cuda()
            else:
                dec_a, dec_b = start[:, :, :n_features//2], start[:, :, n_features//2:]
            dec_a, dec_b = self.proj_dec(dec_a, dec_b)

            # y_a = torch.cat([sos_a, y_a], dim=0)    # add <sos> to front 
            # y_b = torch.cat([sos_b, y_b], dim=0)    # add <sos> to front
            dec_a, dec_b = self.trans_decoder(input_A=dec_a, input_B=dec_b, enc_A=h_as, enc_B=h_bs) 
            y_a, y_b = dec_a, dec_b

            for i in range(max_len - 1):
                dec_a, dec_b = self.trans_decoder(input_A=y_a, input_B=y_b, enc_A=h_as, enc_B=h_bs)
                #print(dec_a[-1].mean().item(), dec_b[-1].mean().item())
                y_a, y_b = torch.cat([y_a, dec_a[-1].unsqueeze(0)], dim=0), torch.cat([y_b, dec_b[-1].unsqueeze(0)], dim=0)
                # dec_a, dec_b = out_as[-1].unsqueeze(0), out_bs[-1].unsqueeze(0)

            # out_ls = out_ls[:-1]  # no need to slice if we <sos> to front and truncate last 
            # out_as = out_as[:-1]

            out_concat = torch.cat([y_a, y_b], dim=-1)
            
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(out_concat))))

        else:
            print("Only one of y and max_len should be input.")
            assert False


        return output
