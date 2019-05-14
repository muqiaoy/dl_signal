import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # TODO key_padding_mask

        # print("[MultiheadAttention forward]")
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        # print("qkv_same: %s" % (str(qkv_same))) 
        # print("kv_same: %s" % (str(kv_same))) 

        # print("query.size(): %s" % (str(query.size())))
        # print("key.size(): %s" % str(key.size())) 
        # print("value.size(): %s" % str(value.size()))

        aved_state = None

        # projecting q, k, v 
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        # scaling q 
        q *= self.scaling

        # extending k, v by one time step at the end, with self.bias_k and self.bias_v 
        if self.bias_k is not None:
            assert self.bias_v is not None
            # print("<self.bias_k not None>")
            # print("k.size(): %s" % str(k.size()))
            # print("v.size(): %s" % str(v.size()))
            # k.size(): torch.Size([30, 256, 32])
            # v.size(): torch.Size([30, 256, 32])

            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

            # print("self.bias_k.size(): %s" % str(self.bias_k.size()))
            # print("self.bias_v.size(): %s" % str(self.bias_v.size()))
            # print("self.bias_k.repeat(1, bsz, 1).size() %s" % str(self.bias_k.repeat(1, bsz, 1).size()))
            # print("k.size(): %s" % str(k.size()))
            # print("v.size(): %s" % str(v.size()))
            # self.bias_k.size(): torch.Size([1, 1, 32])
            # self.bias_v.size(): torch.Size([1, 1, 32])
            # self.bias_k.repeat(1, bsz, 1).size() torch.Size([1, 256, 32])
            # k.size(): torch.Size([31, 256, 32])
            # v.size(): torch.Size([31, 256, 32])

            if attn_mask is not None:
                print("self.bias_k not None") 
                print("attn_mask.shape")
                print(attn_mask.shape) 
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            # k.size(): torch.Size([2048, 31, 4]), head_dim=4, num_heads=8
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            # v.size(): torch.Size([2048, 31, 4]), head_dim=4, num_heads=8

        src_len = k.size(1)

        # extending k, v by another time step at the end, (bsz * num_heads, 1, head_dim) of zeros 
        if self.add_zero_attn:
            print("< self.add_zero_attn > ") 
            # print("k.size(): %s" % str(k.size()))
            # print("v.size(): %s" % str(v.size()))
            # k.size(): torch.Size([2048, 31, 4])
            # v.size(): torch.Size([2048, 31, 4])

            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            # print("k.size(): %s" % str(k.size()))
            # print("v.size(): %s" % str(v.size()))
            # print("k.size(0): %d" % k.size(0))
            # print("v.size(0): %d" % v.size(0))
            # print("k.size()[2:]: %s" % str(k.size()[2:]))
            # print("k.size()[2:]: %s" % str(k.size()[2:]))
            # k.size(): torch.Size([2048, 32, 4])
            # v.size(): torch.Size([2048, 32, 4])
            # k.size(0): 2048, bsz * num_heads
            # v.size(0): 2048, bsz * num_heads 
            # k.size()[2:]: torch.Size([4])
            # k.size()[2:]: torch.Size([4])
            if attn_mask is not None:
                print("self.add_zero_attn")
                print("attn_mask.shape")
                print(attn_mask.shape) 
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        # q: (bsz * num_heads, tgt_len, head_dim)
        # k.transpose(1, 2): (bsz * num_heads, head_dim, src_len + 2) 

        if attn_mask is not None:
            try:
                # print("attn_mask not None")
                # print("attn_weights.shape") 
                # print(attn_weights.shape) 
                # print("attn_mask.shape")
                # print(attn_mask.shape) 
                # # print(attn_mask) 
                # print('attn_mask.unsqueeze(0).shape')
                # print(attn_mask.unsqueeze(0).shape)
                # print("attn_weights.shape")
                # print(attn_weights.shape)

                # attn_weights += attn_mask.unsqueeze(0)
                attn_weights *= attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        # attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = (attn_weights - torch.min(attn_weights)) / (torch.max(attn_weights) - torch.min(attn_weights))
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        # attn_weights: (bsz * num_heads, tgt_len, src_len + 2) 
        # v: (bsz * num_heads, src_len + 2, head_dim) 

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)