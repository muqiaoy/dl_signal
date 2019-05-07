import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder

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
        [self.orig_d_l, self.orig_d_a] = input_dims
        self.input_dim = self.orig_d_l + self.orig_d_a 
        # [self.input_dim] = input_dims 
        # assert self.orig_d_l == self.orig_d_a
        # self.d_l, self.d_a = self.orig_d_l, self.orig_d_a

        self.ntokens = ntokens
        # final_out = self.d_l + self.d_a 
        # final_out = (self.d_l + self.d_a) * time_step 
        # final_out = (self.d_l + self.d_a) * horizons
        final_out = self.input_dim * horizons 
        h_out = hidden_size
#         output_dim = 1
        self.num_heads = num_heads
        self.layers = layers
        self.horizons = horizons
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask

        self.crossmodal = crossmodal
        
        # Transformer networks
        # self.trans = nn.ModuleList([self.get_network() for i in range(self.horizons)])
        self.trans = self.get_network() 
            
        # Projection layers
        # self.proj_l = nn.ModuleList([nn.Linear(self.orig_d_l, self.d_l) for i in range(self.horizons)])
        
        # self.proj_a = nn.ModuleList([nn.Linear(self.orig_d_a, self.d_a) for i in range(self.horizons)])
        # self.proj = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for i in range(self.horizons)]) 
        self.proj = nn.Linear(self.input_dim, self.input_dim)
        
        # self.proj = nn.Linear(final_out, final_out) # Not in the diagram 
        self.out_fc1 = nn.Linear(final_out, h_out)
        
        self.out_fc2 = nn.Linear(h_out, output_dim)
        
        self.out_dropout = nn.Dropout(0.5)
    def get_network(self):
        
        return TransformerEncoder(embed_dim=self.input_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask, crossmodal=self.crossmodal)
            
    def forward(self, x):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).
        """
        # _, batch_size, _ = x.shape
        # x_l = x[:, :, :self.orig_d_l]
        # x_a = x[:, :, self.orig_d_l: self.orig_d_l + self.orig_d_a]
        # print(x.shape) 
        # exit()

        # x_l, x_a = [self.proj_l[i](x_l) for i in range(self.horizons)], [self.proj_a[i](x_a) for i in range(self.horizons)]
        # x = [self.proj[i](x) for i in range(self.horizons)]
        x = self.proj(x)

        # Pass the input through individual transformers
        # h_ls_as = [self.trans[i](x_l[i], x_a[i]) for i in range(self.horizons)] 
        # h = [self.trans[i](x) for i in range(self.horizons)] 
        h = self.trans(x)

         # concat last time steps of all horizons 
        # h_ls_as_each_catted = [torch.cat([h_ls_as[i][0][-1], h_ls_as[i][1][-1]], dim=-1) for i in range(self.horizons)]
        # h_concat = torch.cat(h_ls_as_each_catted, dim=-1)
        h_concat = h[-1] 

        # print("h.shape")
        # print(h.shape)
        # print("h_concat.shape") 
        # print(h_concat.shape)

        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))

        return output, h_concat
