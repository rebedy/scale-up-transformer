import math
from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from axial_positional_embedding import AxialPositionalEmbedding

from transformer_pytorch.reversible import ReversibleSequence, SequentialSequence
from transformer_pytorch.attention import Attention
from transformer_pytorch.favor import FAVORAttention, CrossAttention, ProjectionUpdater


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True).detach()
        return x / maxes

# https://arxiv.org/abs/2103.17239
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

# layer norm


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn, sandwich = False):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         x = self.norm(x)
#         x = self.fn(x, **kwargs)
#         return self.norm_out(x)

class  PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn  # fn: SelfAttention or Chunk
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)  # torch.Size([b, 256, 4000]) -> 2 * torch.Size([b, 256, 2000])
        return x * F.gelu(gates)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    
# class FeedForward(nn.Module):
#     def __init__(self, dim, dropout = 0., mult = 4.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim*mult),# dim * mult * 2),
#             # GEGLU(),
#             GELU(), 
#             nn.Dropout(dropout),
#             nn.Linear(dim * mult, dim)
#         )

#     def forward(self, x, **kwargs):
#         return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)  # 보통 activation = None이라 nn.GELU가 사용됨

        self.glu = glu  # 보통 True
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):  # x: [B, seq_len/self.chunks, dim]  kwargs: {}
        if not self.glu: ###GELU  # 보통 True  
            x = self.w1(x)
            x = self.act(x)
        else:  ### GEGLU
            x, v = self.w1(x).chunk(2, dim=-1)  # -> [B, seq_len/self.chunks, 4*2*dim] -> [B, seq_len/self.chunks, 4*dim]씩 쪼개짐
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)  # -> [B, seq_len/self.chunks, dim]
        return x  # [B, seq_len/self.chunks, dim]


class Chunk(nn.Module):  # for문을 도는 대신 메모리를 적게 써서 FF layer를 통과시키겠다는 의도.
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim  # eg. 1
        self.chunks = chunks  # eg. 10
        self.fn = fn          # FeedForward

    def forward(self, x, **kwargs):  # x: [B, seq_len, dim]   kwargs: {}
        if self.chunks == 1:  # eg. self.chunks = 10  (10조각 내겠다)
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)  # chunks조각 냄. [B, seq_len/self.chunks, dim]이 self.chunks개가 있음
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim) # [B, seq_len/self.chunks, dim] -> [B, seq_len/self.chunks, dim]를 self.chunks조각 모아 concat


# token shifting helper and classes

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)


# positional embeddings

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

# rotary positional embedding(From Rofomer) helpers

def rotate_every_two(x):  # x: [B, global_heads, seq_len, dim_head] q와 k를 의미.
    x = rearrange(x, '... (d j) -> ... d j', j = 2) # -> [B, global_heads, seq_len, dim_head/2, 2]
    x1, x2 = x.unbind(dim = -1) # -> [B, global_heads, seq_len, dim_head/2]
    x = torch.stack((-x2, x1), dim = -1) # -> [B, global_heads, seq_len, dim_head/2, 2]    # x를 절반으로 자르고, 뒤에 chunk에 -를 붙이고 앞으로 옮긴다.
    return rearrange(x, '... d j -> ... (d j)')  # -> [B, global_heads, seq_len, dim_head]

def apply_rotary_pos_emb(q, k, sinu_pos):  # q, k: global head의 q, k [B, global_heads, seq_len, dim_head]    sinu_pos: [1, seq_len, dim_head] 
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)  # [1, seq_len, dim_head] ->  [seq_len, 2, dim_head/2]
    sin, cos = sinu_pos.unbind(dim = -2)  # sin, cos: [seq_len, dim_head/2] Removes a tensor dimension. Returns a tuple of all slices along a given dimension, already without it.
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos)) # -> [seq_len, dim_head]
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k)) # -> [B, global_heads, seq_len, dim_head]
    return q, k

# sinusoidal positional embeddings
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)        # [max_seq_len, dim/2]  outer product
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1) # [max_seq_len, dim]
        self.register_buffer('emb', emb)

    def forward(self, x): # x: [B, seq_len, dim]
        return self.emb[None, :x.shape[1], :].to(x)



#!# main transformer class
 
    
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        condition_len = 0,
        ff_mult = 4,
        nb_features = None,
        chunk_size=128, 
        feature_redraw_interval = 1000,
        reversible = False,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False,
        
        attn_type = None,
        stable_softmax=False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)  # eg. (4, )
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads  # eg. (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:  # 보통 use_scalenorm = False
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:   # 보통 use_rezero = False
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)  # eg. dim=512

        
        self.attn_type = attn_type 
        
        for _, local_heads in zip(range(depth), local_attn_heads):
            if self.attn_type == 'full':
                attn = Attention(
                    dim=dim, 
                    causal = causal, 
                    heads = heads, 
                    dim_head = dim_head, 
                    dropout = attn_dropout,
                    stable_softmax = stable_softmax)
            elif attn_type == 'perf':
                attn = FAVORAttention(dim=dim, 
                                            causal = causal, 
                                            condition_len = condition_len, 
                                            heads = heads, 
                                            dim_head = dim_head, 
                                            local_heads = local_heads, 
                                            local_window_size = local_window_size, 
                                            nb_features = nb_features, 
                                            chunk_size= chunk_size, 
                                            generalized_attention = generalized_attention, 
                                            kernel_fn = kernel_fn, 
                                            dropout = attn_dropout, 
                                            no_projection = no_projection, 
                                            qkv_bias = qkv_bias, 
                                            attn_out_bias = attn_out_bias
                                            )
            ff =  FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))
            
            if not cross_attend:
                continue ## if MLM -> Crosss Attention
            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu))
            ]))


        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)  # len(): 2*depth if cross_attend else depth
        route_context = ((False, False), (True, False)) * depth  # eg. depth = 12 => len(route_context) = 24. 나의 경우 쓸모 없음
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}  # cross attention 안 하는 거면 {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})  # eg. ReversibleSequence으로 감싸진 모델

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw  # auto_check_redraw = True
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):  # x: [B, seq_len, dim]   kwargs = {'pos_emb': [1, seq_len, head_dim], 'mask': [B, seq_len]}
        if self.attn_type == 'perf' and self.auto_check_redraw:   # auto_check_redraw = True
            self.proj_updater.redraw_projections()  # calls_since_last_redraw가 interval을 넘어가면 redraw
        return self.net(x, **kwargs) # -> [B, seq_len, dim]


class TransformerLM(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        num_tokens,
        dim,
        depth,
        heads,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        condition_len = 0,
        ff_mult = 4,
        nb_features = None,
        chunk_size = 128,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(inplace = True), # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False,
        shift_tokens = False,
        
        attn_types=None,
        stable_softmax=False,
        
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)  # eg. 4 -> (4, )
        self.max_seq_len = max_seq_len
        if not chunk_size:
            chunk_size = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        if rotary_position_emb:  # 보통 rotary_position_emb = True
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, 
                                   condition_len, ff_mult, nb_features, chunk_size, feature_redraw_interval, 
                                   reversible, generalized_attention, kernel_fn, use_scalenorm, use_rezero, 
                                   ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, 
                                   qkv_bias, attn_out_bias, shift_tokens, attn_types, stable_softmax)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, **kwargs):  # kwargs = {'mask': tensor with same shape x}
        b, n, device = *x.shape, x.device # b: batch_size, n: x의 seq_len
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embeddings
        x = self.token_emb(x)  # [B, seq_len] -> [B, seq_len, dim]
        x += self.pos_emb(x)   # [B, seq_len, dim]에 [1, seq_len, dim]이 더해짐

        x = self.dropout(x)    # [B, seq_len, dim]

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)  # [1, seq_len, dim_head]
        x = self.transformer(x, pos_emb = layer_pos_emb, **kwargs) # x: [B, seq_len, dim] -> [B, seq_len, dim]

        # norm and to logits
        x = self.norm(x)
        
        if return_encodings:   # 보통 False
            return x

        if exists(self.to_out):
            return self.to_out(x)  # -> [B, seq_len, num_tokens]

        return x @ self.token_emb.weight.t()  # weight tieing했을 시




    
    
"""
class TransformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        reversible = False,
        causal = True,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        nb_features = None,
        chunk_size=None, 
        emb_dropout = 0.1,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(inplace = True), # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        no_projection = False,
        tie_embed = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_types = None,
        stable_softmax = False,
        sandwich_norm = False,
        shift_tokens = False,
        rotary_emb = True
    ):
        # sparse_attn = False,
        super().__init__()
        if not chunk_size:
            chunk_size = max_seq_len
        
        self.max_seq_len=max_seq_len
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        if rotary_emb:  # 보통 rotary_position_emb = True
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)
        
        self.dropout = nn.Dropout(emb_dropout)
        
        layers = nn.ModuleList([])
        # sparse_layer = cast_tuple(sparse_attn, depth)

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        # for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
        for ind, attn_type in zip(range(depth), attn_type_layer):
            if attn_type == 'full':
                attn_class = Attention(
                    dim=dim, 
                    causal = causal, 
                    heads = heads, 
                    dim_head = dim_head, 
                    dropout = attn_dropout,
                    stable_softmax = stable_softmax)
                
            elif attn_type == 'perf':
                attn_class = FAVORAttention(dim=dim, 
                                            causal = causal, 
                                            heads = heads, 
                                            dim_head = dim_head, 
                                            nb_features = nb_features, 
                                            chunk_size= chunk_size, 
                                            generalized_attention = generalized_attention, 
                                            kernel_fn = kernel_fn, 
                                            dropout = attn_dropout, 
                                            no_projection = no_projection, 
                                            qkv_bias = qkv_bias, 
                                            attn_out_bias = attn_out_bias,
                                            )
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')
            
            self.attn_type = attn_type
            attn = attn_class

            ff = FeedForward(dim, mult = ff_mult, dropout = ff_dropout)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))


            layers.append(nn.ModuleList([
                PreLayerNorm(dim, attn, sandwich = sandwich_norm),
                PreLayerNorm(dim, ff, sandwich = sandwich_norm)
            ]))
            #     LayerScale(dim, ind + 1, PreNorm(dim, attn, sandwich = sandwich_norm)),
            #     LayerScale(dim, ind + 1, PreNorm(dim, ff, sandwich = sandwich_norm))
            # ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn}

        self.layers = execute_type(layers, args_route = attn_route_map)
        
        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw  # auto_check_redraw = True
        self.proj_updater = ProjectionUpdater(self.layers, feature_redraw_interval)
        
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None
        
    def forward(self, x, return_encodings = False, **kwargs):
        # text; token and positional embeddings
        x = self.token_emb(x)  # [B, text_seq_len] -> [B, text_seq_len, dim]
        x += self.pos_emb(x)   # [B, text_seq_len, dim]에 [1, text_seq_len, dim]이 더해짐
        
        x = self.dropout(x)    # [B, seq_len, dim]
        
        if self.attn_type == 'perf' and self.auto_check_redraw:   # auto_check_redraw = True
            self.proj_updater.redraw_projections()  # calls_since_last_redraw가 interval을 넘어가면 redraw
        
        x = self.transformer
        layer_pos_emb = self.layer_pos_emb(x)  # [1, seq_len, dim_head]  # TODO: rotary pos emb를 쓴다면 text에 대해서만 적용되어야 함. 가능한가?
        x = self.layers(x, pos_emb = layer_pos_emb, **kwargs) #rotary_pos_emb = layer_pos_emb, 
       
        x = self.norm(x)
        
        if return_encodings:   # 보통 False
            return x

        if exists(self.to_out):
            return self.to_out(x)  # -> [B, seq_len, num_tokens]

        return x @ self.token_emb.weight.t()  # weight tieing했을 시 
    
"""  
    
