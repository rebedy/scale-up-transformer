import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

from functools import partial
from itertools import islice, cycle
from einops import rearrange

from dalle_pytorch.attention import Attention, SparseAxialCausalAttention, SparseConvCausalAttention

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,)*depth


class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


# FeedForward
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*mult*2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*mult, dim)
        )
    def forward(self, x):
        return self.net(x)


# Transformer
class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            seq_len,
            reversible=False,
            causal=True,
            heads=8,
            dim_head=64,
            ff_mult=4,
            attn_dropout=0.,
            ff_dropout=0.,
            attn_types=None,
            image_fmap_size=None,
            sparse_attn=False,
            stable=False,
            sandwich_norm=False,
            shift_tokens=False,
            rotary_emb=True
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)  # TODO: 이게 뭐지?

        for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
            if attn_type == 'full':
                attn_class = partial(Attention, stable=stable)  # 그냥 causal attention
            elif attn_type == 'axial_row':
                attn_class = partial(SparseAxialCausalAttention, seq_len=seq_len, axis=0, imag_size=image_fmap_size, stable=stable)
            elif attn_type == 'axial_col':
                attn_class = partial(SparseAxialCausalAttention, seq_len=seq_len, axis=1, image_size=image_fmap_size, stable=stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len=seq_len, image_size=image_fmap_size, stable=stable)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            attn = attn_class(dim, causal=causal, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout)

            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

            layers.append(nn.ModuleList([
               LayerScale(dim, ind+1, PreNorm(dim, attn, sandwich=sandwich_norm)),
                LayerScale(dim, ind+1, PreNorm(dim, ff, sandwich=sandwich_norm))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),)*depth
        attn_route_map = {'mask':route_attn, 'rotary_pos_emb': route_attn}

        self.layers = execute_type(layers, args_route=attn_route_map)

        pos_emb = None
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb=self.pos_emb, **kwargs)
