import torch
import math
from torch import nn
import torch.nn.functional as F
from operator import mul
from functools import reduce

from performer_pytorch.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

# constant

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def merge_dims(ind_from, ind_to, tensor): # ind_from = 0, ind_to = 1, tensor = [B, 4, windows, window_size, window_size*3]
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]  # shape[arr_slice]: [B, 4]  reduce(mul, shape[arr_slice]): B*4
    return tensor.reshape(*shape)  # [B*4, windows, window_size, window_size*3]

def expand_dim(t, dim, k, unsqueeze=True):  # t = mask[B, windows, window_size, window_size*3]    dim = 1   k = local_heads
    if unsqueeze: # True
        t = t.unsqueeze(dim) # -> [B, 1, windows, window_size, window_size*3]
    expand_shape = [-1] * len(t.shape) # [-1, -1, -1, -1, -1]
    expand_shape[dim] = k # -> [-1, 4, -1, -1, -1]
    return t.expand(*expand_shape) # [B, 1, windows, window_size, window_size*3] -> [B, 4, windows, window_size, window_size*3]

def pad_to_multiple(tensor, multiple, dim=-1, value=0):  # eg. tensor(q, k, v): [B*local_heads, seq_len, dim_head]   dim = -2  # multiple = local_window_size  eg. 256
    seqlen = tensor.shape[dim] 
    m = seqlen / multiple
    if m.is_integer():
        return tensor  # 보통은 m이 정수라 그대로 return됨
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2): # eg. x: [B*local_heads, windows, window_size, dim_head]  backward = 1, forward = 1
    t = x.shape[1]   # eg. 2
    dims = (len(x.shape) - dim) * (0, 0)  # eg. (0, 0, 0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)  # -> [B*local_heads, backward+windows+forward, window_size, dim_head]
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]  # pad한 차원에서 sliding하면서 tensor 저장. [B*local_heads, windows, window_size, dim_head]가 3개 담겨 있음
    return torch.cat(tensors, dim=dim) # [B*local_heads, windows, window_size, dim_head]가 3개 담겨 있음 -> [B*local_heads, windows, window_size*3, dim_head]

# main class

class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)  # causal이면 0, bi면 1
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.window_size = window_size   # local_window_size  eg. 256
        self.causal = causal
        self.look_backward = look_backward  # 1
        self.look_forward = look_forward    # causal이면 0, bi면 1
        self.exact_windowsize = exact_windowsize # False
        self.autopad = autopad  # 보통 True

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk  # False

        self.rel_pos = None
        if exists(rel_pos_emb_config) or exists(dim):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):  # rel_pos_emb_config: (dim_head, local_heads) eg. (64, 4)
                dim = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim)

    def forward(self, q, k, v, input_mask = None):  # q, k, v: [B, local_heads, seq_len, dim_head]  input_mask: [B, seq_len]
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v)) # -> q, k, v: [B*local_heads, seq_len, dim_head] 

        if exists(self.rel_pos):
            pos_emb = self.rel_pos(q)  # -> [1, seq_len, dim_head]
            q, k = apply_rotary_pos_emb(q, k, pos_emb) # -> q, k: [B*local_heads, seq_len, dim_head]

        if self.autopad: # 보통 True
            orig_t = q.shape[1]
            q, k, v = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))  # -> [B*local_heads, seq_len, dim_head] # seq_len이 window_size로 나눠지면 그대로 return됨.

        window_size, causal, look_backward, look_forward, shared_qk = self.window_size, self.causal, self.look_backward, self.look_forward, self.shared_qk
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        assert (t % window_size) == 0, f'sequence length {t} must be divisible by window size {window_size} for local attention'

        windows = t // window_size

        if shared_qk:  # shared_qk = False
            k = F.normalize(k, 2, dim=-1).type_as(q)

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :] # [1, seq_len]
        b_t = ticker.reshape(1, windows, window_size)  # -> [1, windows, window_size]

        bucket_fn = lambda t: t.reshape(b, windows, window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))  # [B*local_heads, seq_len, dim_head] -> [B*local_heads, windows, window_size, dim_head]

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward} # 보통 {'backward': 1, 'forward': 1}
        bk = look_around(bk, **look_around_kwargs) # -> [B*local_heads, windows, window_size*3, dim_head]  'backward': 1, 'forward': 1인 경우.
        bv = look_around(bv, **look_around_kwargs) # -> [B*local_heads, windows, window_size*3, dim_head]  'backward': 1, 'forward': 1인 경우.

        bq_t = b_t  # [1, windows, window_size]
        bq_k = look_around(b_t, **look_around_kwargs) # -> [1, windows, window_size*3] 'backward': 1, 'forward': 1인 경우.

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)  # -> [B*local_heads, windows, window_size, window_size*3]

        mask_value = max_neg_value(dots)

        if shared_qk: # shared_qk = False
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                mask = mask | (bq_t[:, :, :, None] > (bq_k[:, :, None, :] + max_causal_window_size))

            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1  # look_around에서 패드부분이 True가 됨
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            if self.autopad:  # True
                input_mask = pad_to_multiple(input_mask, window_size, dim=-1, value=False)  # -> seq_len이 window_size로 나눠지면 그대로 return됨. [B, seq_len] 
            input_mask = input_mask.reshape(-1, windows, window_size)  # -> [B, windows, window_size]
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs) # -> mk: [B, windows, window_size*3] 'backward': 1, 'forward': 1인 경우.
            mask = (mq[:, :, :, None] * mk[:, :, None, :])  # -> [B, windows, window_size, 1] * [B, windows, 1, window_size*3] = [B, windows, window_size, window_size*3]
            mask = merge_dims(0, 1, expand_dim(mask, 1, h)) # -> [B*4, windows, window_size, window_size*3]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv) # -> [B*local_heads, windows, window_size, dim_head]
        out = out.reshape(-1, t, e) # -> [B*local_heads, seq_len, dim_head]

        if self.autopad:
            out = out[:, :orig_t, :] # -> [B*local_heads, seq_len, dim_head]

        return out.reshape(*shape) # -> [B, local_heads, seq_len, dim_head]