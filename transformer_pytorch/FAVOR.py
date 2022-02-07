import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, repeat
import math
from functools import partial

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from transformer_pytorch.model_utils import *


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None): # projection_matrix: [nb_features, dim_heads]. chunk마다는 orthogonal하지만 다른 chunk의 vector와는 not orthogonal.
    b, h, *_ = data.shape  # data: q 또는 k: [B, global_head, seq_len, dim_head]

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.  # NOTE: 왜 ** -0.25이지? ** -0.5가 아니라?  왜 1/sqrt(d)가 아니냐는 말임. 그 이유는, query쪽에서 한 번, key쪽에서 한 번 해줘서 둘을 곱하면 ** -0.5가 됨. 

    ratio = (projection_matrix.shape[0] ** -0.5)   # paper에서 1/sqrt(m)을 의미하는 것 같음

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h) # -> [B, global_head, nb_features, dim_heads]  projection_matrix를 batch차원, global_head차원으로 복제.
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection) # -> [B, global_head, seq_len, nb_features]  # 수식에서 wTx에 해당    data(q 또는 k) 와 projection을 inner product

    diag_data = data ** 2 # [B, global_head, seq_len, dim_head]
    diag_data = torch.sum(diag_data, dim=-1)  # [B, global_head, seq_len]
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2) # [B, global_head, seq_len]  # 수식에서 exp 안에 ||x||^2 / 2 에 해당.   # data(Q 또는 K)에 1/sqrt(sqrt(d))로 normalize먼저 한 후 제곱하고 더한 후 2로 나누는 것과 동일.
    diag_data = diag_data.unsqueeze(dim=-1)  # [B, global_head, seq_len, 1]  

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -                                   # torch.max(data_dash, dim=-1, keepdim=True).values: [B, global_head, seq_len, 1]
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)   # -> [B, global_head, seq_len, nb_features]
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)      # -> [B, global_head, seq_len, nb_features]
    # NOTE: exp 안에 max값을 빼는 이유: Numerical stability. 나중에 attention계산시 분모에도 Q', K'이 들어가므로 최종 결과에는 영향 없음
    return data_dash.type_as(data) # [B, global_head, seq_len, nb_features]  data(Q 또는 K)의 random feature vector를 얻었다. 즉 Q', K'에 해당

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)




def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.qr(unstructured_block.cpu(), some = True)  # q, r: [cols, cols]  QR decomposition. input = QR with Q being an orthogonal matrix or batch of orthogonal matrices and R being an upper triangular matrix or batch of upper triangular matrices.
    q, r = map(lambda t: t.to(device), (q, r))              # torch.mm(q.t(), q).round() = I
    return q.t()  # -> row벡터들이 서로 orthogonal

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):   # nb_rows로 nb_features을 받고, nb_columns로 dim_heads를 받음
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)  # -> row벡터들이 서로 orthogonal. [nb_columns, nb_columns]. 자기 자신의 norm은 1
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)  # NOTE: 왜 이렇게 하나? dim_heads차원에서 만들 수 있는 orthogonal vector의 개수는 최대 dim_heads개이다. 따라서 dim_heads개씩 orthogonal vector set을 만들어 두는 것.
    # 따라서 torch.mm(final_matrix, final_matrix.t())해보면 chunk마다는 orthogonal하지만 다른 chunk의 vector와는 not orthogonal
    if scaling == 0:   # 보통 0
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)  # [nb_rows]   row별 L2-norm. tensor([ 7.9014,  7.9234,  7.0152,  6.4810,  7.9664,  8.0006,...])
    elif scaling == 1:  # NOTE: regularized softmax-kernel(SMREG)인 듯
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)  # [nb_rows]   원소들이 전부 qsrt(nb_columns). tensor([ 8., 8., ...])
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix  # [nb_rows, nb_rows]  @ [nb_rows, nb_columns] = [nb_rows, nb_columns]
              # 즉, scaling == 0이면 orthogonal한 random feature들의 길이가 다 다른거고, scaling == 1이면 길이가 qsrt(nb_columns)로 전부 동일한 것(구 표현에 다 있다는 것). 

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):  # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    k_lensum = k.sum(dim = -2)  # -> [B, global_head, nb_features]  논문의 식에서 (K')T 1L 에 해당. 
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_lensum.type_as(q))  # [B, global_head, seq_len] 논문의 식에서 Q' ((K')T 1L)에 해당.
    context = torch.einsum('...nd,...ne->...de', k, v) # -> [B, global_head, nb_features, dim_head]   n에 해당하는 차원(seq_len) 으로 sum됨. 논문의 식에서 (K')T V
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv) # -> [B, global_head, seq_len, dim_head]   # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 그리고 row별(query별)로 D_inv값 곱함.
    return out  # [B, global_head, seq_len, dim_head]  

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6): # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    last_k_cumsum = 0        # 마지막 Q' ((K')T 1L) 를 의미.
    last_context_cumsum = 0  # 마지막 (K')T V 를 의미.
    outs = []
    # chunk화해서 for문 돌리는 이유: 메모리를 좀 더 써서 for문을 적게 돌겠다.
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))): #q,k:[B, global_head, seq_len/chunk_size, nb_features]  v:[B, global_head, seq_len/chunk_size, dim_head]
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2) # [B, global_head, seq_len/chunk_size, nb_features]  논문의 식에서 (K')T 1L 에 해당. 다만 causal때문에 cumsum임에 주의. 층위구조.

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps) # -> [B, global_head, seq_len/chunk_size]  논문의 식에서 Q' ((K')T 1L)에 해당. 다만 층위구조.
        context = torch.einsum('...nd,...ne->...nde', k, v)  # -> [B, global_head, seq_len/chunk_size, nb_features ,dim_head] outer product.  논문의 식에서 (K')T V.  다만 같은 seq 위치별로.
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)  # [B, global_head, seq_len/chunk_size, nb_features ,dim_head]
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv) # -> [B, global_head, seq_len/chunk_size, dim_head] # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 다만 query별로. 그리고 row별(query별)로 D_inv값 곱함.

        last_k_cumsum = k_cumsum[:, :, -1:] # [B, global_head, 1, nb_features]  # 이건 다음 for문(다음 chunk)에서 D_inv를 구하기 위해 필요. 마지막것만 가져와도 이미 cumsum을 한 상태이기 때문에 괜찮다.
        last_context_cumsum = context_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features ,dim_head]  # 이건 다음 for문(다음 chunk)에서 context_cumsum를 구하기 위해 필요. 마지막것만 가져와도 이미 cumsum을 한 상태이기 때문에 괜찮다.
        outs.append(out)

    return torch.cat(outs, dim = -2)  # -> [B, global_head, seq_len, dim_head]

def conditioned_causal_linear_attention_noncuda(q, k, v, condition_len, chunk_size = 32, eps = 1e-6): # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    condition_q = q[:, :, :condition_len]  # [B, global_head, condition_len, nb_features]
    condition_k = k[:, :, :condition_len]  # [B, global_head, condition_len, nb_features]
    condition_v = v[:, :, :condition_len]  # [B, global_head, condition_len, dim_head]

    condition_k_cumsum = condition_k.sum(dim = -2) + eps  # -> [B, global_head, nb_features]  논문의 식에서 (K')T 1L 에 해당. 
    condition_D_inv = 1. / torch.einsum('...nd,...d->...n', condition_q, condition_k_cumsum.type_as(condition_q))  # [B, global_head, seq_len] 논문의 식에서 Q' ((K')T 1L)에 해당.
    condition_context = torch.einsum('...nd,...ne->...de', condition_k, condition_v) # -> [B, global_head, nb_features, dim_head]   n에 해당하는 차원(seq_len) 으로 sum됨. 논문의 식에서 (K')T V
    condition_out = torch.einsum('...de,...nd,...n->...ne', condition_context, condition_q, condition_D_inv) # -> [B, global_head, seq_len, dim_head]   # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 그리고 row별(query별)로 D_inv값 곱함.
    
    last_k_cumsum = condition_k_cumsum.unsqueeze(2)
    last_context_cumsum = condition_context.unsqueeze(2)
    outs = [condition_out]

    # chunk화해서 for문 돌리는 이유: 메모리를 좀 더 써서 for문을 적게 돌겠다.
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q[:, :, condition_len:], k[:, :, condition_len:], v[:, :, condition_len:]))): #q,k:[B, global_head, seq_len/chunk_size, nb_features]  v:[B, global_head, seq_len/chunk_size, dim_head]
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2) # [B, global_head, seq_len/chunk_size, nb_features]  논문의 식에서 (K')T 1L 에 해당. 다만 causal때문에 cumsum임에 주의. 층위구조.

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps) # -> [B, global_head, seq_len/chunk_size]  논문의 식에서 Q' ((K')T 1L)에 해당. 다만 층위구조.
        context = torch.einsum('...nd,...ne->...nde', k, v)  # -> [B, global_head, seq_len/chunk_size, nb_features ,dim_head] outer product.  논문의 식에서 (K')T V.  다만 같은 seq 위치별로.
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)  # [B, global_head, seq_len/chunk_size, nb_features ,dim_head]
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv) # -> [B, global_head, seq_len/chunk_size, dim_head] # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 다만 query별로. 그리고 row별(query별)로 D_inv값 곱함.

        last_k_cumsum = k_cumsum[:, :, -1:] # [B, global_head, 1, nb_features]  # 이건 다음 for문(다음 chunk)에서 D_inv를 구하기 위해 필요.
        last_context_cumsum = context_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features ,dim_head]  # 이건 다음 for문(다음 chunk)에서 context_cumsum를 구하기 위해 필요.
        outs.append(out)

    return torch.cat(outs, dim = -2)  # -> [B, global_head, seq_len, dim_head]

def conditioned_causal_linear_attention(q, k, v, condition_len, chunk_size = 32, eps = 1e-6): # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    
    ## image part
    condition_q = q[:, :, :condition_len]  # [B, global_head, condition_len, nb_features]
    condition_k = k[:, :, :condition_len]  # [B, global_head, condition_len, nb_features]
    condition_v = v[:, :, :condition_len]  # [B, global_head, condition_len, dim_head]

    condition_k_cumsum = condition_k.sum(dim = -2)  # -> [B, global_head, nb_features]  논문의 식에서 (K')T 1L 에 해당. 
    condition_D_inv = 1. / torch.einsum('...nd,...d->...n', condition_q, condition_k_cumsum.type_as(condition_q))  # [B, global_head, seq_len] 논문의 식에서 Q' ((K')T 1L)에 해당.
    condition_context = torch.einsum('...nd,...ne->...de', condition_k, condition_v) # -> [B, global_head, nb_features, dim_head]   n에 해당하는 차원(seq_len) 으로 sum됨. 논문의 식에서 (K')T V
    condition_out = torch.einsum('...de,...nd,...n->...ne', condition_context, condition_q, condition_D_inv) # -> [B, global_head, seq_len, dim_head]   # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 그리고 row별(query별)로 D_inv값 곱함.
    
    last_k_cumsum = condition_k_cumsum.unsqueeze(2)
    last_context_cumsum = condition_context.unsqueeze(2)
    outs = [condition_out]
    
    ## efficient causal linear attention, created by EPFL
    
    from fast_transformers.causal_product import CausalDotProduct
    from fast_transformers.local_product import LocalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)
    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply
    local_dot_product_fn = amp.float_function(LocalDotProduct.apply) if is_half else LocalDotProduct.apply

    q = q[:, :, condition_len:] #q,k:[B, global_head, seq_len/chunk_size, nb_features]
    k = k[:, :, condition_len:]
    v = v[:, :, condition_len:] #  v:[B, global_head, seq_len/chunk_size, dim_head]
    
    k_cumsum = last_k_cumsum + k.cumsum(dim=-2) + eps # [B, global_head, seq_len/chunk_size, nb_features]  논문의 식에서 (K')T 1L 에 해당. 다만 causal때문에 cumsum임에 주의. 층위구조.
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    outs.append(out)

    return torch.cat(outs, dim = -2)  # -> [B, global_head, seq_len, dim_head]


class FastAttention(nn.Module):
    ## generalize kernel은 무시하고 코드 작성
    def __init__(
            self,
            dim_head,
            nb_features = None,
            ortho_scaling = 0,
            causal = False,
            condition_len = 0,
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn = nn.ReLU(),
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

        self.generalized_attention=generalized_attention
        self.kernel_fn = kernel_fn   # 보통 ReLU
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            if attn_type == "noncuda":
                self.causal_linear_fn = partial(causal_linear_attention_noncuda, condition_len=condition_len)
            elif attn_type == "cuda":
                self.causal_linear_fn = partial(causal_linear_attention, condition_len=condition_len)
            if attn_type in ["condi_noncuda", "conditioned_noncuda"]:
                self.causal_linear_fn = partial(conditioned_causal_linear_attention_noncuda, condition_len=condition_len)
            elif attn_type in ["condi_cuda", "conditioned_cuda"]:
                self.causal_linear_fn = partial(conditioned_causal_linear_attention, condition_len=condition_len)
    
    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:  # 보통 False
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)  # -> [B, global_head, seq_len, nb_features] 즉, q의 random feature vector를 얻었다
            k = create_kernel(k, is_query = False) # -> [B, global_head, seq_len, nb_features] 즉, k의 random feature vector를 얻었다

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out  # [B, global_head, seq_len, dim_head]



class FAVORAttention(nn.Module):
    """Cross Attention은 고려하지 않고 코드 작성"""
    def __init__(
            self,
            dim,
            causal = False,
            condition_len = 0,
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn = nn.ReLU(),
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
        self.fast_attention = FastAttention(dim_head, nb_features, 
                                            causal=causal, condition_len=condition_len, attn_type=attn_type,
                                            generalized_attention=generalized_attention,
                                            kernel_fn = kernel_fn,
                                            no_projection=no_projection)

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





