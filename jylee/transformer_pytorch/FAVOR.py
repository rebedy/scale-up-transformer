import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import math
from functools import partial

from transformer_pytorch.model_utils import *

def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None): # nb_rows로 nb_features을 받고, nb_columns로 dim_heads를 받음
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:  # regularized softmax-kernel(SMREG)인 듯
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(data, *, projection_matrix, is_query, normalized_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape  # data는 q 또는 k

    data_normalizer = (data.shape[-1] ** -0.25) if normalized_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5) # paper에서 1/sqrt(m)을 의미하는 것 같음

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h) # -> [B, global_head, nb_features, dim_heads]  projection_matrix를 batch차원, global_head차원으로 복제.
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)  # -> [B, global_head, seq_len, nb_features]  # 수식에서 wTx에 해당    data(q 또는 k) 와 projection을 inner product

    diag_data = data ** 2   # [B, global_head, seq_len, dim_head]
    diag_data = torch.sum(diag_data, dim=-1) # [B, global_head, seq_len]
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2) # [B, global_head, seq_len] # 수식에서 exp 안에 ||x||^2 / 2 에 해당.   # data(Q 또는 K)에 1/sqrt(sqrt(d))로 normalize먼저 한 후 제곱하고 더한 후 2로 나누는 것과 동일.
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)



def conditioned_causal_linear_attention_noncuda(q, k, v, condition_len, chunk_size=128, eps=1e-6):
    """
    q, k : [B, global_head, seq_len, nb_features]
    v : [B, global_head, seq_len, dim_head]
    """
    condition_q = q[:, :, :condition_len]
    condition_k = k[:, :, :condition_len]
    condition_v = v[:, :, :condition_len]

    condition_k_cumsum = condition_k.sum(dim=-2)   # -> [B, global_head, nb_features]   논문에서 (K')T 1L에 해당
    condition_D_inv = 1. / torch.einsum('...nd,...d->...n', condition_q, condition_k_cumsum.type_as(
        condition_q))  # [B, global_head, seq_len] 논문의 식에서 Q' ((K')T 1L)에 해당.
    condition_context = torch.einsum('...nd,...ne->...de', condition_k,
                                     condition_v)  # -> [B, global_head, nb_features, dim_head]   n에 해당하는 차원(seq_len) 으로 sum됨. 논문의 식에서 (K')T V
    condition_out = torch.einsum('...de,...nd,...n->...ne', condition_context, condition_q,
                                 condition_D_inv)  # -> [B, global_head, seq_len, dim_head]   # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 그리고 row별(query별)로 D_inv값 곱함.

    last_k_cumsum = condition_k_cumsum.unsqueeze(2)
    last_context_cumsum = condition_context.unsqueeze(2)
    outs = [condition_out]

    # chunk화해서 for문 돌리는 이유: 메모리를 좀 더 써서 for문을 적게 돌겠다.
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q[:, :, condition_len:], k[:, :, condition_len:],
                                                                    v[:, :,
                                                                    condition_len:]))):  # q,k:[B, global_head, seq_len/chunk_size, nb_features]  v:[B, global_head, seq_len/chunk_size, dim_head]
        k_cumsum = last_k_cumsum + k.cumsum(
            dim=-2)  # [B, global_head, seq_len/chunk_size, nb_features]  논문의 식에서 (K')T 1L 에 해당. 다만 causal때문에 cumsum임에 주의. 층위구조.

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(
            q) + eps)  # -> [B, global_head, seq_len/chunk_size]  논문의 식에서 Q' ((K')T 1L)에 해당. 다만 층위구조.
        context = torch.einsum('...nd,...ne->...nde', k,
                               v)  # -> [B, global_head, seq_len/chunk_size, nb_features ,dim_head] outer product.  논문의 식에서 (K')T V.  다만 같은 seq 위치별로.
        context_cumsum = last_context_cumsum + context.cumsum(
            dim=-3)  # [B, global_head, seq_len/chunk_size, nb_features ,dim_head]
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q,
                           D_inv)  # -> [B, global_head, seq_len/chunk_size, dim_head] # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 다만 query별로. 그리고 row별(query별)로 D_inv값 곱함.

        last_k_cumsum = k_cumsum[:, :,
                        -1:]  # [B, global_head, 1, nb_features]  # 이건 다음 for문(다음 chunk)에서 D_inv를 구하기 위해 필요.
        last_context_cumsum = context_cumsum[:, :,
                              -1:]  # [B, global_head, 1, nb_features ,dim_head]  # 이건 다음 for문(다음 chunk)에서 context_cumsum를 구하기 위해 필요.
        outs.append(out)

    return torch.cat(outs, dim=-2)  # -> [B, global_head, seq_len, dim_head]



class FastAttention(nn.Module):
    ## generalize kernel은 무시하고 코드 작성
    def __init__(
            self,
            dim_head,
            nb_features = None,
            ortho_scaling = 0,
            causal = False,
            condition_len = 0,
            no_projection = False
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_head * math.log(dim_head)))

        self.dim_head = dim_head
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_head, scaling=ortho_scaling)
        projection_matrix = self.create_projection() # [nb_features, dim_heads]. chunk마다는 orthogonal하지만 다른 chunk의 vector와는 not orthogonal. ortho_scaling=0이면 가우시안에서 임의로 [nb_features, dim_heads]을 만들고 norm(dim=1)한 값인 [nb_features]를 row별로 곱한다.
        self.register_buffer('projection_matrix', projection_matrix)

        self.no_projection = no_projection

        self.causal = causal
        if causal:
            self.causal_linear_fn = partial(conditioned_causal_linear_attention_noncuda, condition_len=condition_len)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:   # TODO: 이게 뭐지?
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)  # -> [B, global_head, seq_len, nb_features] 즉, q의 random feature vector를 얻었다
            k = create_kernel(k, is_query=False)  # -> [B, global_head, seq_len, nb_features] 즉, k의 random feature vector를 얻었다

        attn_fn = self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out  # [B, global_head, seq_len, dim_head]

class FAVORAttention(nn.Module):
    """Cross Attention은 고려하지 않고 코드 작성"""
    def __init__(
            self,
            dim,
            causal = False,
            condition_len = 0,
            heads = 8,
            local_heads = 0,
            nb_features = None,
            dropout = 0.3,
            no_projection = False,
            qkv_bias = False,
            attn_out_bias = True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = (dim // heads)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.dim_head = dim_head

        # Fast Attention
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, condition_len=condition_len, no_projection=no_projection)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_output = nn.Linear(dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        context = default(context, x)
        context_mask = default(context_mask, mask)

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  # => q, k, v: [B, head, seq_len, dim_head]
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        if not empty(q):  # global attention이 존재한다면
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]  # [B, 1, seq_len, 1]
                v.masked_fill(~global_mask, 0.)

            if exists(pos_emb):
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = self.fast_attention(q, k, v) # [B, global_head, seq_len, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')  # -> [B, seq_len, inner_dim]
        out = self.to_output(out)  # -> [B, seq_len, dim]
        return self.dropout(out)  # -> [B, seq_len, dim]


class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance  # ReversibleSequence로 감싸진 모델
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval: # calls_since_last_redraw가 interval을 넘어가면 redraw
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)  # list
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented





