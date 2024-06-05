import torch
import torch.nn.functional as F

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