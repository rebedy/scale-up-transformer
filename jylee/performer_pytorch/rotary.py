import torch
from torch import nn, einsum

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # size(): [dim_head/2]
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):  # x: [B*local_heads, seq_len, dim_head]
        n = x.shape[-2]
        t = torch.arange(n, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq) # outer product  [seq_len, dim_head/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # 아직 sin, cos은 안 취하고 일단 합쳐서 return함. rotary pos emb 적용하려고.
        return emb[None, :, :]  # [1, seq_len, dim_head]

def rotate_half(x):
    x = x.reshape((x.shape[0], -1, 2, x.shape[-1] // 2))
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k