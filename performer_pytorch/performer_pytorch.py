import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

from performer_pytorch.local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from performer_pytorch.reversible import ReversibleSequence, SequentialSequence
from performer_pytorch.filter_logits_functions import top_k, top_p

try:
    from apex import amp
    APEX_AVAILABLE = True

except ImportError:
    APEX_AVAILABLE = False


# helpers
def eval_decorator(fn):
    def inner(model, *args, **kwargs):  # args: (text[:1]), kwargs: {'filter_thres': 0.9}
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)  # generated image: [1, 3, 256, 256]
        model.train(was_training)  # model.training을 was_training값으로 바꿈
        return out
    return inner


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):  # projection_matrix: [nb_features, dim_heads]. chunk마다는 orthogonal하지만 다른 chunk의 vector와는 not orthogonal.
    b, h, *_ = data.shape  # data: q 또는 k: [B, global_head, seq_len, dim_head]

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.  # NOTE: 왜 ** -0.25이지? ** -0.5가 아니라?  왜 1/sqrt(d)가 아니냐는 말임. 그 이유는, query쪽에서 한 번, key쪽에서 한 번 해줘서 둘을 곱하면 ** -0.5가 됨.

    ratio = (projection_matrix.shape[0] ** -0.5)   # paper에서 1/sqrt(m)을 의미하는 것 같음

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)  # -> [B, global_head, nb_features, dim_heads]  projection_matrix를 batch차원, global_head차원으로 복제.
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)  # -> [B, global_head, seq_len, nb_features]  # 수식에서 wTx에 해당    data(q 또는 k) 와 projection을 inner product

    diag_data = data ** 2  # [B, global_head, seq_len, dim_head]
    diag_data = torch.sum(diag_data, dim=-1)  # [B, global_head, seq_len]
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)  # [B, global_head, seq_len]  # 수식에서 exp 안에 ||x||^2 / 2 에 해당.   # data(Q 또는 K)에 1/sqrt(sqrt(d))로 normalize먼저 한 후 제곱하고 더한 후 2로 나누는 것과 동일.
    diag_data = diag_data.unsqueeze(dim=-1)  # [B, global_head, seq_len, 1]

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
        # torch.max(data_dash, dim=-1, keepdim=True).values: [B, global_head, seq_len, 1]
        # -> [B, global_head, seq_len, nb_features]
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)      # -> [B, global_head, seq_len, nb_features]
    # NOTE: exp 안에 max값을 빼는 이유: Numerical stability. 나중에 attention계산시 분모에도 Q', K'이 들어가므로 최종 결과에는 영향 없음
    return data_dash.type_as(data)  # [B, global_head, seq_len, nb_features]  data(Q 또는 K)의 random feature vector를 얻었다. 즉 Q', K'에 해당


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)  # q, r: [cols, cols]  QR decomposition. input = QR with Q being an orthogonal matrix or batch of orthogonal matrices and R being an upper triangular matrix or batch of upper triangular matrices.
    q, r = map(lambda t: t.to(device), (q, r))              # torch.mm(q.t(), q).round() = I
    return q.t()  # -> row벡터들이 서로 orthogonal


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):   # nb_rows로 nb_features을 받고, nb_columns로 dim_heads를 받음
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)  # -> row벡터들이 서로 orthogonal. [nb_columns, nb_columns]. 자기 자신의 norm은 1
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)  # NOTE: 왜 이렇게 하나? dim_heads차원에서 만들 수 있는 orthogonal vector의 개수는 최대 dim_heads개이다. 따라서 dim_heads개씩 orthogonal vector set을 만들어 두는 것.
    # 따라서 torch.mm(final_matrix, final_matrix.t())해보면 chunk마다는 orthogonal하지만 다른 chunk의 vector와는 not orthogonal
    if scaling == 0:   # 보통 0
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)  # [nb_rows]   row별 L2-norm. tensor([ 7.9014,  7.9234,  7.0152,  6.4810,  7.9664,  8.0006,...])
    elif scaling == 1:  # NOTE: regularized softmax-kernel(SMREG)인 듯
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)  # [nb_rows]   원소들이 전부 qsrt(nb_columns). tensor([ 8., 8., ...])
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix  # [nb_rows, nb_rows]  @ [nb_rows, nb_columns] = [nb_rows, nb_columns]
# 즉, scaling == 0이면 orthogonal한 random feature들의 길이가 다 다른거고, scaling == 1이면 길이가 qsrt(nb_columns)로 전부 동일한 것(구 표현에 다 있다는 것).


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):  # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    k_cumsum = k.sum(dim=-2)  # -> [B, global_head, nb_features]  논문의 식에서 (K')T 1L 에 해당.
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # [B, global_head, seq_len] 논문의 식에서 Q' ((K')T 1L)에 해당.
    context = torch.einsum('...nd,...ne->...de', k, v)  # -> [B, global_head, nb_features, dim_head]   n에 해당하는 차원(seq_len) 으로 sum됨. 논문의 식에서 (K')T V
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)  # -> [B, global_head, seq_len, dim_head]   # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 그리고 row별(query별)로 D_inv값 곱함.
    return out  # [B, global_head, seq_len, dim_head]


# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps=1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled=False)

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
def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-6):  # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    last_k_cumsum = 0        # 마지막 Q' ((K')T 1L) 를 의미.
    last_context_cumsum = 0  # 마지막 (K')T V 를 의미.
    outs = []
    # chunk화해서 for문 돌리는 이유: 메모리를 좀 더 써서 for문을 적게 돌겠다.
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):  # q,k:[B, global_head, seq_len/chunk_size, nb_features]  v:[B, global_head, seq_len/chunk_size, dim_head]
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)  # [B, global_head, seq_len/chunk_size, nb_features]  논문의 식에서 (K')T 1L 에 해당. 다만 causal때문에 cumsum임에 주의. 층위구조.

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)  # -> [B, global_head, seq_len/chunk_size]  논문의 식에서 Q' ((K')T 1L)에 해당. 다만 층위구조.
        context = torch.einsum('...nd,...ne->...nde', k, v)  # -> [B, global_head, seq_len/chunk_size, nb_features ,dim_head] outer product.  논문의 식에서 (K')T V.  다만 같은 seq 위치별로.
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)  # [B, global_head, seq_len/chunk_size, nb_features ,dim_head]
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)  # -> [B, global_head, seq_len/chunk_size, dim_head]  # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 다만 query별로. 그리고 row별(query별)로 D_inv값 곱함.

        last_k_cumsum = k_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features]  # 이건 다음 for문(다음 chunk)에서 D_inv를 구하기 위해 필요.
        last_context_cumsum = context_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features ,dim_head]  # 이건 다음 for문(다음 chunk)에서 context_cumsum를 구하기 위해 필요.
        outs.append(out)

    return torch.cat(outs, dim=-2)  # -> [B, global_head, seq_len, dim_head]


def conditioned_causal_linear_attention_noncuda(q, k, v, condition_len, chunk_size=128, eps=1e-6):  # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    condition_q = q[:, :, :condition_len]  # [B, global_head, condition_len, nb_features]
    condition_k = k[:, :, :condition_len]  # [B, global_head, condition_len, nb_features]
    condition_v = v[:, :, :condition_len]  # [B, global_head, condition_len, dim_head]

    condition_k_cumsum = condition_k.sum(dim=-2)  # -> [B, global_head, nb_features]  논문의 식에서 (K')T 1L 에 해당.
    condition_D_inv = 1. / torch.einsum('...nd,...d->...n', condition_q, condition_k_cumsum.type_as(condition_q))  # [B, global_head, seq_len] 논문의 식에서 Q' ((K')T 1L)에 해당.
    condition_context = torch.einsum('...nd,...ne->...de', condition_k, condition_v)  # -> [B, global_head, nb_features, dim_head]   n에 해당하는 차원(seq_len) 으로 sum됨. 논문의 식에서 (K')T V
    condition_out = torch.einsum('...de,...nd,...n->...ne', condition_context, condition_q, condition_D_inv)  # -> [B, global_head, seq_len, dim_head]   # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 그리고 row별(query별)로 D_inv값 곱함.

    last_k_cumsum = condition_k_cumsum.unsqueeze(2)
    last_context_cumsum = condition_context.unsqueeze(2)
    outs = [condition_out]

    # chunk화해서 for문 돌리는 이유: 메모리를 좀 더 써서 for문을 적게 돌겠다.
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q[:, :, condition_len:], k[:, :, condition_len:], v[:, :, condition_len:]))):  # q,k:[B, global_head, seq_len/chunk_size, nb_features]  v:[B, global_head, seq_len/chunk_size, dim_head]
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)  # [B, global_head, seq_len/chunk_size, nb_features]  논문의 식에서 (K')T 1L 에 해당. 다만 causal때문에 cumsum임에 주의. 층위구조.

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)  # -> [B, global_head, seq_len/chunk_size]  논문의 식에서 Q' ((K')T 1L)에 해당. 다만 층위구조.
        context = torch.einsum('...nd,...ne->...nde', k, v)  # -> [B, global_head, seq_len/chunk_size, nb_features ,dim_head] outer product.  논문의 식에서 (K')T V.  다만 같은 seq 위치별로.
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)  # [B, global_head, seq_len/chunk_size, nb_features ,dim_head]
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)  # -> [B, global_head, seq_len/chunk_size, dim_head]  # 논문의 식에서 Q'와 (K')T V를 메트릭스 곱한 것임. 다만 query별로. 그리고 row별(query별)로 D_inv값 곱함.

        last_k_cumsum = k_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features]  # 이건 다음 for문(다음 chunk)에서 D_inv를 구하기 위해 필요.
        last_context_cumsum = context_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features ,dim_head]  # 이건 다음 for문(다음 chunk)에서 context_cumsum를 구하기 위해 필요.
        outs.append(out)

    return torch.cat(outs, dim=-2)  # -> [B, global_head, seq_len, dim_head]


class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, condition_len=0, generalized_attention=False, kernel_fn=nn.ReLU(), no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads          # eg. 64
        self.nb_features = nb_features      # eg. 256
        self.ortho_scaling = ortho_scaling  # eg. 0   (0 또는 1)

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling)
        # 레이어별로 projection_matrix를 만들어 놓는구나
        projection_matrix = self.create_projection()  # [nb_features, dim_heads]. chunk마다는 orthogonal하지만 다른 chunk의 vector와는 not orthogonal. ortho_scaling=0이면 가우시안에서 임의로 [nb_features, dim_heads]을 만들고 norm(dim=1)한 값인 [nb_features]를 row별로 곱한다.
        self.register_buffer('projection_matrix', projection_matrix)  # 즉, scaling == 0이면 orthogonal한 random feature들의 길이가 다 다른거고, scaling == 1이면 길이가 qsrt(nb_columns)로 전부 동일한 것(구 표현에 다 있다는 것).

        self.generalized_attention = generalized_attention  # 보통 False. NOTE: 이게 의미하는 바가 뭐지?
        self.kernel_fn = kernel_fn   # 보통 ReLU

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection  # 보통 False

        self.causal=causal
        if causal:
            self.causal_linear_fn = partial(conditioned_causal_linear_attention_noncuda, condition_len=condition_len)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):  # q, k, v: [B, global_head, seq_len, dim_head]
        device = q.device

        if self.no_projection:  # 보통 False
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn, projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)  # -> [B, global_head, seq_len, nb_features] 즉, q의 random feature vector를 얻었다
            k = create_kernel(k, is_query=False)  # -> [B, global_head, seq_len, nb_features] 즉, k의 random feature vector를 얻었다

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)  # -> [B, global_head, seq_len, dim_head]
        return out  # [B, global_head, seq_len, dim_head]


# a module for keeping track of when to update the projections

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

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:  # calls_since_last_redraw가 interval을 넘어가면 redraw
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)  # list
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplementedError


# classes

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


class Chunk(nn.Module):  # for문을 도는 대신 메모리를 적게 써서 FF layer를 통과시키겠다는 의도.
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim  # eg. 1
        self.chunks = chunks  # eg. 10
        self.fn = fn          # FeedForward

    def forward(self, x, **kwargs):  # x: [B, seq_len, dim]   kwargs: {}
        if self.chunks == 1:  # eg. self.chunks = 10  (10조각 내겠다)
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)  # chunks조각 냄. [B, seq_len/self.chunks, dim]이 self.chunks개가 있음
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)  # [B, seq_len/self.chunks, dim] -> [B, seq_len/self.chunks, dim]를 self.chunks조각 모아 concat


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)  # 보통 activation = None이라 nn.GELU가 사용됨

        self.glu = glu  # 보통 True
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):  # x: [B, seq_len/self.chunks, dim]  kwargs: {}
        if not self.glu:  # 보통 True
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)  # -> [B, seq_len/self.chunks, 4*2*dim] -> [B, seq_len/self.chunks, 4*dim]씩 쪼개짐
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)  # -> [B, seq_len/self.chunks, dim]
        return x  # [B, seq_len/self.chunks, dim]


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        condition_len=0,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)  # dim_head가 존재하면 그걸 그대로 return
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, condition_len=condition_len, generalized_attention=generalized_attention, kernel_fn=kernel_fn, no_projection=no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout, look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)  # 보통은 dim과 inner_dim같게 함. qkv_bias = False
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)  # 보통은 attn_out_bias = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):  # x: [B, seq_len, dim]  pos_emb: [1, seq_len, dim_head]  mask: [B, seq_len]
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads   # eg. self.heads = 8, self.global_heads = 4

        cross_attend = exists(context)  # 보통 False

        context = default(context, x)  # context가 없으면 x를 return. =>  context: [B, seq_len, dim]
        context_mask = default(context_mask, mask) if not cross_attend else context_mask  # context_mask가 없으면 mask를 return. => context_mask: [B, seq_len]

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)  # => q, k, v: [B, seq_len, inner_dim] 근데 보통 inner_dim == dim

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  # => q, k, v: [B, head, seq_len, dim_head]
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))  # 앞의 head개수 중 global_heads개는 global head로 사용하고 나머지는 local head로 사용
        # q, k, v: [B, global_heads, seq_len, dim_head]  lq, lk, lv: [B, local_heads, seq_len, dim_head]
        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]  # -> [B, 1, seq_len, 1]
                v.masked_fill_(~global_mask, 0.)   # NOTE: 이런 방식은 arbitrary한 마스크는 못 만들지만 모든 query가 특정 key를 보지 못하게는 만들 수 있음. 근데 그럼 attn score는 sum to 1인데 v에는 score의 leaking발생.

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)  # positional embedding으로 rotary positional emb.  layer마다 적용된다! NOTE: text token들에 대해서만 적용하려면 여기를 수정해야 하는데 새로 클래스 만들어서 수정하기!

            out = self.fast_attention(q, k, v)  # -> [B, global_head, seq_len, dim_head]  여기가 핵심! 
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)  # -> [B, local_heads, seq_len, dim_head]
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)  # -> [B, heads, seq_len, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')  # -> [B, seq_len, inner_dim]
        out = self.to_out(out)   # -> [B, seq_len, dim]
        return self.dropout(out)  # -> [B, seq_len, dim]


class SelfAttention(Attention):  # 여기에서 __init__()을 정의하지 않은 것은, Attention에 있는 __init__()을 덮어쓰지 않고 그대로 쓰겠다는 것
    def forward(self, *args, context=None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)  # args: (tensor([[[-0.5840, -...='cuda:0'),)    # kargs: {'pos_emb': tensor([[[ 0.0000,  ...='cuda:0'), 'mask': tensor([[True, True,...='cuda:0')}


class CrossAttention(Attention):
    def forward(self, *args, context=None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context=context, **kwargs)


# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


# rotary positional embedding(From Rofomer) helpers

def rotate_every_two(x):  # x: [B, global_heads, seq_len, dim_head] q와 k를 의미.
    x = rearrange(x, '... (d j) -> ... d j', j=2)  # -> [B, global_heads, seq_len, dim_head/2, 2]
    x1, x2 = x.unbind(dim=-1)  # -> [B, global_heads, seq_len, dim_head/2]
    x = torch.stack((-x2, x1), dim=-1)  # -> [B, global_heads, seq_len, dim_head/2, 2]    # x를 절반으로 자르고, 뒤에 chunk에 -를 붙이고 앞으로 옮긴다.
    return rearrange(x, '... d j -> ... (d j)')  # -> [B, global_heads, seq_len, dim_head]


def apply_rotary_pos_emb(q, k, sinu_pos):  # q, k: global head의 q, k [B, global_heads, seq_len, dim_head]    sinu_pos: [1, seq_len, dim_head]
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)  # [1, seq_len, dim_head] ->  [seq_len, 2, dim_head/2]
    sin, cos = sinu_pos.unbind(dim=-2)  # sin, cos: [seq_len, dim_head/2] Removes a tensor dimension. Returns a tuple of all slices along a given dimension, already without it.
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2), (sin, cos))  # -> [seq_len, dim_head]
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))  # -> [B, global_heads, seq_len, dim_head]
    return q, k


# sinusoidal positional embeddings

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)        # [max_seq_len, dim/2]  outer product
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)  # [max_seq_len, dim]
        self.register_buffer('emb', emb)

    def forward(self, x):  # x: [B, seq_len, dim]
        return self.emb[None, :x.shape[1], :].to(x)


# performer

class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        condition_len=0,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        ff_glu=False,
        ff_dropout=0.,
        attn_dropout=0.,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
        attn_out_bias=True
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

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([  # f, g
                wrapper_fn(SelfAttention(dim, causal=causal, condition_len=condition_len, heads=heads, dim_head=dim_head,
                                         local_heads=local_heads, local_window_size=local_window_size, nb_features=nb_features,
                                         generalized_attention=generalized_attention, kernel_fn=kernel_fn, dropout=attn_dropout,
                                         no_projection=no_projection, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads=heads, dim_head=dim_head, nb_features=nb_features, generalized_attention=generalized_attention,
                                          kernel_fn=kernel_fn, dropout=attn_dropout, no_projection=no_projection, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)  # len(): 2*depth if cross_attend else depth
        route_context = ((False, False), (True, False)) * depth  # eg. depth = 12 => len(route_context) = 24. 나의 경우 쓸모 없음
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}  # cross attention 안 하는 거면 {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})  # eg. ReversibleSequence으로 감싸진 모델

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw  # auto_check_redraw = True
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):  # x: [B, seq_len, dim]   kwargs = {'pos_emb': [1, seq_len, head_dim], 'mask': [B, seq_len]}
        if self.auto_check_redraw:   # auto_check_redraw = True
            self.proj_updater.redraw_projections()  # calls_since_last_redraw가 interval을 넘어가면 redraw
        return self.net(x, **kwargs)  # -> [B, seq_len, dim]


class PerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head=64,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        condition_len=0,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.,
        ff_dropout=0.,
        attn_dropout=0.,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        tie_embed=False,
        rotary_position_emb=True,
        axial_position_emb=False,
        axial_position_shape=None,
        auto_check_redraw=True,
        qkv_bias=False,
        attn_out_bias=False
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)  # eg. 4 -> (4, )

        self.max_seq_len = max_seq_len
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

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, condition_len, ff_mult,
                                   nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn,
                                   use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection,
                                   auto_check_redraw, qkv_bias, attn_out_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, **kwargs):  # kwargs = {'mask': tensor with same shape x}
        b, n, device = *x.shape, x.device  # b: batch_size, n: x의 seq_len
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embeddings
        x = self.token_emb(x)  # [B, seq_len] -> [B, seq_len, dim]
        x += self.pos_emb(x)   # [B, seq_len, dim]에 [1, seq_len, dim]이 더해짐

        x = self.dropout(x)    # [B, seq_len, dim]

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)  # [1, seq_len, dim_head]
        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)  # x: [B, seq_len, dim] -> [B, seq_len, dim]

        # norm and to logits
        x = self.norm(x)

        if return_encodings:   # 보통 False
            return x

        if exists(self.to_out):
            return self.to_out(x)  # -> [B, seq_len, num_tokens]

        return x @ self.token_emb.weight.t()  # weight tieing했을 시


class PerformerLM_i2t(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,      # text vocab size
        num_img_tokens,  # img vocab size + num img pad
        max_seq_len,     # total max len; img_len * max_img_num + max_text_len
        max_img_num,     # num img slot
        dim,
        depth,
        heads,
        dim_head=64,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        condition_len=0,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.,
        ff_dropout=0.,
        attn_dropout=0.,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        tie_embed=False,
        rotary_position_emb=True,
        axial_position_emb=False,
        axial_position_shape=None,
        auto_check_redraw=True,
        qkv_bias=False,
        attn_out_bias=False,
        img_fmap_size=0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_img_num = max_img_num
        self.condition_len = condition_len
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dim = dim

        # img;
        if condition_len != 0:
            self.image_token_emb = nn.Embedding(num_img_tokens, dim)
            self.image_pos_emb = AxialPositionalEmbedding(dim=dim, axial_shape=(img_fmap_size, img_fmap_size))

        # text;
        self.token_emb = nn.Embedding(num_tokens, dim)   # num_tokens = text vocab size
        if rotary_position_emb:
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

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, condition_len, ff_mult,
                                   nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn,
                                   use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection,
                                   auto_check_redraw, qkv_bias, attn_out_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, images, texts, return_encodings=False, **kwargs):  # kwargs = {'mask': tensor with same shape x}
        b, n_img, device = *images.shape, images.device  # b: batch_size, n_img: image의 tot_seq_len
        b, n_txt, device = *texts.shape, texts.device  # b: batch_size, n_txt: text의 tot_seq_len
        n = n_img + n_txt
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        assert n_img == self.condition_len, f'image length {n_img} must be equal to the condition length {self.condition_len}'

        # img; token and positional embeddings
        if self.condition_len == 0:
            x_img = torch.empty(b, n_img, self.dim).to(device)
        else:
            x_img = self.image_token_emb(images)  # -> [B, tot_img_len, dim]
            outs = []
            for x_img_slot in x_img.chunk(self.max_img_num, dim=1):  # x_img_slot: [B, img_len, dim]
                out = self.image_pos_emb(x_img_slot)  # out: [B, img_len, dim]
                outs.append(out)
            x_img_pos = torch.cat(outs, dim=1)  # -> [B, tot_img_len, dim]
            x_img += x_img_pos

        # text; token and positional embeddings
        x_text = self.token_emb(texts)  # [B, text_seq_len] -> [B, text_seq_len, dim]
        x_text += self.pos_emb(x_text)   # [B, text_seq_len, dim]에 [1, text_seq_len, dim]이 더해짐

        # merge
        x = torch.cat((x_img, x_text), dim=1)    # -> [B, seq_len, dim]

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)  # [1, seq_len, dim_head]  # TODO: rotary pos emb를 쓴다면 text에 대해서만 적용되어야 함. 가능한가?
        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)  # x: [B, seq_len, dim] -> [B, seq_len, dim]

        # norm and to logits
        x = self.norm(x)

        if return_encodings:   # 보통 False
            return x

        if exists(self.to_out):
            return self.to_out(x)  # -> [B, seq_len, num_tokens]

        return x @ self.token_emb.weight.t()  # weight tieing했을 시

    @torch.no_grad()
    @eval_decorator
    def generate_texts(
        self,
        images,   # tensor[B, tot_img_len]
        *,
        sos_token_idx=None,
        eos_token_idx=None,
        pad_token_idx=None,
        filter_logits_fn='top_k',
        filter_thres=0.9,
        temperature=1.,
    ):
        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be in (top_k, top_p)')

        B, image_seq_len, device = *images.shape, images.device
        total_len = self.max_seq_len

        # [B, image_seq_len+1]에서 시작. 점점 길어질 것임.
        out = torch.cat((images, torch.tensor([[sos_token_idx]] * B).to(device)), dim=-1)

        for cur_len in range(image_seq_len + 1, total_len):

            image, text = out[:, :image_seq_len], out[:, image_seq_len:]

            logits = self(image, text)[:, -1, :]  # -> logits: [B, num_text_tokens]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)  # [B, num_text_tokens]
            sample = torch.multinomial(probs, 1)  # [B, 1]

            out = torch.cat((out, sample), dim=-1)

            # break check
            if ((out[:, image_seq_len:] == eos_token_idx).sum(dim=-1) > 0).sum() == B:
                break

        text_seq = out[:, image_seq_len:]  # [B, <text_seq_len]

        # postprocess
        indices = [list(row).index(eos_token_idx) if eos_token_idx in row else -1 for row in text_seq]
        for row, idx in enumerate(indices):
            if idx >= 0:
                text_seq[row, idx + 1:] = pad_token_idx
        return text_seq
