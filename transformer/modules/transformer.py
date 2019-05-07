import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
        crossmodal (boo): whether we do cross-modal transformer or not
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, attn_mask=False, crossmodal=None):
        super().__init__()
        self.dropout = 0      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        # self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        
        self.crossmodal = crossmodal
        
        self.attn_mask = attn_mask


        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input_A, input_B):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim) if cross-modal`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim) if cross-modal`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        input_A = self.scale_embed_position_dropout(input_A)
        input_B = self.scale_embed_position_dropout(input_B)
        # For each transformer encoder layer:
        for layer in self.layers:
            input_A, input_B = layer(input_A, input_B)
        return input_A, input_B

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training) # may change
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Hubert
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            add_bias_kv=False, 
            add_zero_attn=False
        )
        self.attn_mask = attn_mask
        self.crossmodal = True
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1_A = Linear(self.embed_dim, 8*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2_A = Linear(8*self.embed_dim, self.embed_dim)
        self.fc1_B = Linear(self.embed_dim, 8*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2_B = Linear(8*self.embed_dim, self.embed_dim)

        self.layer_norms_A = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])
        self.layer_norms_B = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x_A, x_B):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        ## Attention Part
        # Residual and Layer Norm
        residual_A = x_A
        residual_B = x_B
        # Multihead Attention
        x_aaa = self.attention_block(x_A, x_A, x_A)
        x_aab = self.attention_block(x_A, x_A, x_B)
        x_aba = self.attention_block(x_A, x_B, x_A)
        x_baa = self.attention_block(x_B, x_A, x_A)
        x_abb = self.attention_block(x_A, x_B, x_B)
        x_bab = self.attention_block(x_B, x_A, x_B)
        x_bba = self.attention_block(x_B, x_B, x_A)
        x_bbb = self.attention_block(x_B, x_B, x_B)


        
        x_A = x_aaa - x_abb - x_bab - x_bba
        x_B = -x_bbb + x_baa + x_aba + x_aab

        x_A = self.layer_norms_A[0](x_A)
        x_B = self.layer_norms_B[0](x_B)
        # Dropout and Residual
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        

        # print('r_A', sum(residual_A))
        # print(sum(x_A))
        # print('x_A', sum(self.layer_norms_A[2](x_A)))
        # print('r_B', sum(residual_B))
        # print('x_B', sum(self.layer_norms_B[2](x_B)))
        # assert False
        x_A = residual_A + x_A
        x_B = residual_B + x_B
        
        # x_A = self.layer_norms_A[0](x_A)
        # x_B = self.layer_norms_B[0](x_B)
        
        
        ##FC Part
        residual_A = x_A
        residual_B = x_B
        
        # FC1
        x_A = F.relu(self.fc1_A(x_A))
        x_B = F.relu(self.fc1_B(x_B))
        x_A = F.dropout(x_A, p=self.relu_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.relu_dropout, training=self.training)
        
        # FC2
        x_A = self.fc2_A(x_A)
        x_B = self.fc2_B(x_B)

        x_A = self.layer_norms_A[1](x_A)
        x_B = self.layer_norms_B[1](x_B)

        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        
        x_A = residual_A + x_A
        x_B = residual_B + x_B

        # x_A = self.layer_norms_A[1](x_A)
        # x_B = self.layer_norms_B[1](x_B)

        return x_A, x_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training) # may change
        return x

    def attention_block(self, x, x_k, x_v):

        mask = None

#         x   = self.maybe_layer_norm(0, x, before=True)
#         x_k = self.maybe_layer_norm(0, x_k, before=True)
#         x_v = self.maybe_layer_norm(0, x_v, before=True) 
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m





if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)

