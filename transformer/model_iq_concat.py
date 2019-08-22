import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules.transformer_concat import TransformerEncoder, TransformerDecoder
from models import *
from utils import count_parameters
from conv import Conv1d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, ntokens, time_step, input_dims, hidden_size, embed_dim, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False, crossmodal=False):
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
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            )
        [self.orig_d_a, self.orig_d_b] = input_dims
        assert self.orig_d_a == self.orig_d_b
        channels = ((((((((((self.orig_d_a + self.orig_d_b -6)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 
            -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1 -3)//1+1 -2)//2+1
        self.d_x = channels * 128
        self.ntokens = ntokens
        final_out = embed_dim
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask
        self.embed_dim = embed_dim
        self.crossmodal = crossmodal
        
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
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask, crossmodal=self.crossmodal)
            
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
