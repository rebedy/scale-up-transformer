import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

from math import log2, sqrt
import numpy as np
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return self.val

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class DALLE(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            dim,
            depth,
            heads,
            dim_head,
            reversible=False,
            attn_dropout=0.0,
            ff_dropout=0.0,
            sparse_attn=False,   ## TODO: 이게 뭐지?
            attn_types=None,
            loss_img_weight=7,
            ff_mult=4,
            sandwich_norm=False,
            shift_tokens=True,
            rotary_position_emb=True,
            axial_position_emb=False
    ):
        super().__init__()
        