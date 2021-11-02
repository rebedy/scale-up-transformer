import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def empty(tensor):
    return tensor.numel() == 0

def eval_decorator(fn):
    def inner(model, *args, **kwargs):  # args: (text[:1]), kwargs: {'filter_thres': 0.9}
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs) # generated image: [1, 3, 256, 256]
        model.train(was_training)  # model.training을 was_training값으로 바꿈
        return out
    return

def top_k(logits, thres = 0.9): # eg. logits: [B, num_tokens], thres=0.9
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1) # eg. k = 884
    val, ind = torch.topk(logits, k) # eg. val: [B, 884]  ind: [B, 884]
    probs = torch.full_like(logits, float('-inf')) # logits과 같은 크기의 tensor가 나오는데 원소가 전부 float('-inf')로 채워져 있음
    probs.scatter_(1, ind, val)   # 나머지는 다 -inf가 된 상황
    return probs # [B, num_tokens]

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# Rotary positional embedding
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


# LAyerNorm
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

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn  # fn: SelfAttention or Chunk
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

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



# SequentialSequence

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router] # matched_keys =['mask', 'pos_emb']

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes) # new_f_args: {'pos_emb': tensor[1, seq_len, head_dim]}  new_g_args = {}
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs): # x: tensor[B, seq_len, dim]   kwargs: {'pos_emb': tensor[1, seq_len, dim_head], 'mask': tensor[B, seq_len]}
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


