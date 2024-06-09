
from functools import reduce
import copy
import math
import numpy as np

import torch
from torch.nn import ModuleList
import torch.nn.functional as F


def base2index(base, vocab):
    return vocab[base]


def index2base_batch(base, vocab):
    batch_seq = []
    for b in range(len(base)):
        seq = []
        for b in base[b]:
            for v in vocab:
                if vocab[v] == b:
                    seq.append(v)
        batch_seq.append(seq)
    return batch_seq  # vocab[base]


def index2base(base, vocab):
    seq = []
    for b in base:
        for v in vocab:
            if vocab[v] == b:
                seq.append(v)
    return seq  # vocab[base]


def itos(self, field, batch):  # batch에서 원본 sentence 얻는 함수
    with torch.cuda.device_of(batch):
        batch = batch.tolist()
    batch = [[field.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

    def trim(s, t):  # 현재 token ~ <EOS> token 사이의 문장 return
        sentence = []
        for w in s:
            if w == t:
                break
            sentence.append(w)
        return sentence

    batch = [trim(ex, field.eos_token) for ex in batch]  # batch를 문장으로

    def filter_special(tok):
        return tok not in (field.init_token, field.pad_token)

    batch = [' '.join(list(filter(filter_special, ex))) for ex in batch]
    return batch


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


def clones(module, N):
    """
        ModuleList는 목록에 하위 모듈을 보관. 이때 모듈들은 파이썬 리스트들 처럼 인덱스를 사용할 수 있다.
    """
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def subsequent_mask(size):
    """
    디코더에서 어텐션 스코어 매트릭스에서 이후의 값들에 대해 -∞으로 마스킹 처리해주기 위한 함수.
    (1, size, size)의 마스크를 리턴한다.
    "Mask out subsequent positions."
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def log(t, eps=1e-9):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()
