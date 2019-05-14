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
        # assert False
        x_l, x_a = [self.proj_l[i](x_l) for i in range(self.horizons)], [self.proj_a[i](x_a) for i in range(self.horizons)]

        # Pass the input through individual transformers
        h_ls_as = [self.trans[i](x_l[i], x_a[i]) for i in range(self.horizons)] 
        h_ls_as_each_catted = [torch.cat([h_ls_as[i][0], h_ls_as[i][1]], dim=-1) for i in range(self.horizons)]
        h_concat = torch.cat(h_ls_as_each_catted, dim=-1)
        
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))
        # No sigmoid because we use BCEwithlogitis which contains sigmoid layer and more stabl
        output = output.transpose(0,1)
        return output, h_concat

class TransformerGenerationModel(nn.Module):
    def __init__(self, ntokens, input_dims, hidden_size, output_dim, num_heads, attn_dropout, relu_dropout, res_dropout, layers, horizons, attn_mask=False, src_mask=False, tgt_mask=False, crossmodal=False):
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
        [self.orig_d_l, self.orig_d_a] = input_dims
        assert self.orig_d_l == self.orig_d_a
        self.d_l, self.d_a = self.orig_d_l, self.orig_d_a
        # [self.d_l, self.d_a] = proj_dims

        self.ntokens = ntokens
        # final_out = self.d_l + self.d_a 
        # final_out = (self.d_l + self.d_a) * time_step 
        # final_out = (self.d_l + self.d_a) *  horizons
        final_out = self.d_l 
        h_out = hidden_size
#         output_dim = 1
        self.num_heads = num_heads
        self.layers = layers
        self.horizons = horizons
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask # for encoder 
        # self.src_mask = src_mask  # for decoder
        # self.tgt_mask = tgt_mask  # for decoder

        self.crossmodal = crossmodal
        
        # Transformer networks
        self.trans_encoder = nn.ModuleList([self.get_encoder_network() for i in range(self.horizons)])
        self.trans_decoder = nn.ModuleList([self.get_decoder_network() for i in range(self.horizons)])

        print("Encoder Model size: {0}".format(count_parameters(self.trans_encoder)))
        print("Decoder Model size: {0}".format(count_parameters(self.trans_decoder)))
            
        # Projection layers
        self.proj_l = nn.ModuleList([nn.Linear(self.orig_d_l, self.d_l) for i in range(self.horizons)])
        
        self.proj_a = nn.ModuleList([nn.Linear(self.orig_d_a, self.d_a) for i in range(self.horizons)])
        
        # self.proj = nn.Linear(final_out, final_out) # Not in the diagram 
        self.out_fc1_A = nn.Linear(final_out, h_out)
        self.out_fc1_B = nn.Linear(final_out, h_out)
        
        self.out_fc2_A = nn.Linear(h_out, final_out)
        self.out_fc2_B = nn.Linear(h_out, final_out)
        
        self.out_dropout = nn.Dropout(0.5)

    def get_encoder_network(self):
        
        return TransformerEncoder(embed_dim=self.orig_d_l, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask, crossmodal=self.crossmodal)

    def get_decoder_network(self): 
        return TransformerDecoder(embed_dim=self.orig_d_l, num_heads=self.num_heads, layers=self.layers, src_attn_dropout=self.attn_dropout, 
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, tgt_attn_dropout=self.attn_dropout, crossmodal=self.crossmodal)
            
    def forward(self, x, y):
        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).  
        """
        _, batch_size, _ = x.shape
        x_l = x[:, :, :self.orig_d_l]  
        x_a = x[:, :, self.orig_d_l: self.orig_d_l + self.orig_d_a]
        
        # x_l, x_a = self.proj_l(x_l), self.proj_a(x_a)
        x_l, x_a = [self.proj_l[i](x_l) for i in range(self.horizons)], [self.proj_a[i](x_a) for i in range(self.horizons)]

        # Pass the input through individual transformers
        # h_ls, h_as = self.trans(x_l, x_a)  # Dimension (L, N, C)
        h_ls_as = [self.trans_encoder[i](x_l[i], x_a[i]) for i in range(self.horizons)] 

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
        enc_x_l, enc_x_a = h_ls_as[0] 

        seq_len, batch_size, n_features2 = y.shape 
        n_features = int(n_features2 / 2)

        y_l = y[:-1, :, :self.orig_d_l]                               # truncate last target 
        y_a = y[:-1, :, self.orig_d_l: self.orig_d_l + self.orig_d_a] # truncate last target 
        # print("y_l.shape")
        # print(y_l.shape)
        # print("y_a.shape") 
        # print(y_a.shape)

        # print(y_l.shape)
        sos_l = torch.zeros(1, batch_size, n_features).cuda()
        sos_a = torch.zeros(1, batch_size, n_features).cuda()
        y_l = torch.cat([sos_l, y_l], dim=0)    # add <sos> to front 
        y_a = torch.cat([sos_a, y_a], dim=0)    # add <sos> to front 
 
        # print("y_l.shape")
        # print(y_l.shape)
        # print("y_a.shape") 
        # print(y_a.shape)
        # exit()
        # print("exiting")
        # exit()
        out_ls_as = [self.trans_decoder[i](input_A=y_l, input_B=y_a, enc_A=enc_x_l, enc_B=enc_x_a) for i in range(self.horizons)] 

        # print("exiting")
        # exit()
        out_ls, out_as = out_ls_as[0] 
        # out_ls = out_ls[:-1]  # no need to slice if we <sos> to front and truncate last 
        # out_as = out_as[:-1]

        out_A = self.out_fc2_A(self.out_dropout(F.relu(self.out_fc1_A(out_ls))))
        out_B = self.out_fc2_B(self.out_dropout(F.relu(self.out_fc1_B(out_as))))

        out_concated = torch.cat([out_A, out_B], dim=-1) # (TS, BS, feature_dim) 
        # out_concated = out_concated.transpose(0, 1) # (BS, TS, feature_dim) 

        return out_concated # (TS, BS, feature_dim)  
        # return output, h_ls_as   # (list dim = horizons, tuple dim = 2, Tensor(10 (TS), 256 (BS), 160 (a_dim/b_dim)))
