import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, repeat
import time
import math

from transformer_pytorch.model_utils import *


class TransformerAttention(nn.Module):
    def __init__(
            self,
            dim,  # 모델 안에 들어가는 총 dimension
            causal = False,
            condition_len = 0,
            heads = 8,
            local_heads = 0,
            dropout = 0.3,
            qkv_bias = False,
            attn_out_bias = True,
            **kwargs
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = (dim // heads)  # 각 attention head 마다의 dimension

        self.heads = heads
        self.global_heads = heads - local_heads
        self.dim_head = dim_head
        self.dim = dim

        # Attention
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)


    def multihead_scaled_dot_product(self, query, key, value, mask, causal_mask):
        """
        causal_mask = [seq_len, seq_len]
        """
        seq_len = query.size(2)
        attn_weights = (torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim_head))  # [B, head, seq_len, seq_len]

        # hard coding  ## TODO: change
        causal_mask = torch.ones(seq_len, seq_len).to(attn_weights.device)
        causal_mask[:, :1024] = 0

        small_causal_mask = torch.ones(seq_len-1024, seq_len-1024).to(attn_weights.device)
        small_causal_mask = torch.triu(small_causal_mask, diagonal=1)

        causal_mask[1024:, 1024:] = small_causal_mask

        # causal_mask = torch.triu(causal_mask, diagonal=1)

        # causal mask
        if causal_mask is not None:  # 당연히 None은 아님
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1).bool(), float("-inf"))

        # padding mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        starttime = time.monotonic()
        attn_weights = F.softmax(attn_weights, dim=-1)
        endtime = time.monotonic()
        # print(f'softmax time: {endtime-starttime}')

        starttime = time.monotonic()
        attn_output = torch.matmul(attn_weights, value)
        endtime = time.monotonic()
        # print(f'multiply with key matrix {endtime-starttime}')

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()   # [B, seq_len, head, dim_head]
        concat_attn_output_shape = attn_output.size()[:-2] + (self.dim,)
        attn_output = attn_output.view(*concat_attn_output_shape)
        attn_output = self.to_out(attn_output)
        return attn_output, attn_weights


    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, causal_mask=None, **kwargs):
        """
        x shape: [B, seq_len, dim]
        pos_emb shape: [B, seq_len, dim_head]
        mask shape : [B, seq_len]  여기서 mask는 padding용
        """
        b, n, _ = x.shape
        h = self.heads
        gh = self.global_heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)   # [B, seq_len, dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))   # [B, head, seq_len, dim_head]
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))   #  앞의 head 개수 중 global heads개는 global head로 사용하고 나머지는 local head로 사용

        ## rotary positional embedding
        if exists(pos_emb):
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        attn_output, attn_weights = self.multihead_scaled_dot_product(q, k, v, mask, causal_mask)   # [B, seq_len, dim]
        return self.dropout(attn_output)   # [B, seq_len, dim]