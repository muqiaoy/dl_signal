import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules.transformer import TransformerEncoder

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class TransformerModel(nn.Module):
    def __init__(self, ntokens, time_step, input_dims, hidden_size, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, layers, horizons, attn_mask=False, crossmodal=False):
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
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=6, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(32, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(64, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            Flatten(),
            
            # nn.Linear(256*32, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, output_size),
            # nn.Sigmoid()
            )
        [self.orig_d_l, self.orig_d_a] = input_dims
        assert self.orig_d_l == self.orig_d_a
        self.d_l, self.d_a = 1664//2, 1664//2

        self.ntokens = ntokens
        final_out = (self.orig_d_l + self.orig_d_a) *  horizons
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.horizons = horizons
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask

        self.crossmodal = crossmodal
        
        # Transformer networks
        self.trans = nn.ModuleList([self.get_network() for i in range(self.horizons)])
            
        # Projection layers
        self.proj_l = nn.ModuleList([nn.Linear(self.d_l, self.orig_d_l) for i in range(self.horizons)])
        
        self.proj_a = nn.ModuleList([nn.Linear(self.d_a, self.orig_d_a) for i in range(self.horizons)])
        
        # self.proj = nn.Linear(final_out, final_out) # Not in the diagram 
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(0.5)
    def get_network(self):
        
        return TransformerEncoder(embed_dim=self.orig_d_l, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask, crossmodal=self.crossmodal)
            
    def forward(self, x):
        """
        x should have dimension [batch_size, seq_len, n_features] (i.e., N, L, C).
        """
        batch_size = x.shape[0]
        # print(x.shape)
        x = x.reshape(-1, 2, self.orig_d_l) # （seq_len, 2, n_features）
        x = self.cnn(x)
        # print(x.shape)
        x = x.reshape(batch_size, -1, self.d_l + self.d_a)
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l: self.d_l + self.d_a]
        # print(x_a.shape)
        x_l, x_a = [self.proj_l[i](x_l) for i in range(self.horizons)], [self.proj_a[i](x_a) for i in range(self.horizons)]

        # Pass the input through individual transformers
        h_ls_as = [self.trans[i](x_l[i], x_a[i]) for i in range(self.horizons)] 
        h_ls_as_each_catted = [torch.cat([h_ls_as[i][0], h_ls_as[i][1]], dim=-1) for i in range(self.horizons)]
        h_concat = torch.cat(h_ls_as_each_catted, dim=-1)
        
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))
        # No sigmoid because we use BCEwithlogitis which contains sigmoid layer and more stabl
        output = output.transpose(0,1)
        return output, h_concat
