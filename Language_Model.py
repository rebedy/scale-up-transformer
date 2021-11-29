import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torch.nn as nn
from performer_pytorch import PerformerLM
import time

NUM_TOKENS = 15000
MAX_SEQ_LEN = 4000
BATCH_SIZE = 2

model = PerformerLM(
    num_tokens = NUM_TOKENS,
    max_seq_len = MAX_SEQ_LEN,      # max sequence length   ## reversible = False이면 max_seq_len이 최대 4096, True이면 32배 더 가능
    dim = 512,                      # dimension.  dimension must be divisible by number of heads
    depth = 12,                     # layers
    heads = 8,                      # heads
    dim_head = 64,                  # dim of head    ## inner_dim = dim_head * heads로 결정된다.
    causal = True,                  # auto-regressive or not
    condition_len=0,                # if greater than 1 and causal=True, conditoned causal LM works.
    nb_features = 256,              # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head. 이게 dim_head보다 크면 nb_features/dim_head개의 block orthogonal matirx를 만든다.
    feature_redraw_interval = 1000, # how frequently to redraw the projection matrix, the more frequent, the slower the training
    generalized_attention = False,  # defaults to softmax approximation, but can be set to True for generalized attention
    kernel_fn = nn.ReLU(),          # the kernel function to be used, if generalized attention is turned on, defaults to Relu
    reversible = True,              # reversible layers, from Reformer paper  ## 원래는 True. Performer안에서 정의됨
    ff_chunks = 10,                 # chunk feedforward layer, from Reformer paper
    use_scalenorm = False,          # use scale norm, from 'Transformers without Tears' paper  ## 원래 False
    use_rezero = False,             # use rezero, from 'Rezero is all you need' paper    ## 원래 False.      scalenorm, rezero, layernorm 중 한가지만 사용 가능.
    tie_embed = True,              # multiply final embeddings with token weights for logits, like gpt decoder   ## 원래 False
    ff_glu = True,                  # use GLU variant for feedforward
    emb_dropout = 0.1,              # embedding dropout
    ff_dropout = 0.1,               # feedforward dropout
    attn_dropout = 0.1,             # post-attn dropout
    local_attn_heads = 0,           # n heads are local attention, heads-n others are global performers. lucid가 만든 것 같음. local_attn_heads = 0이 진정한 performer.
    local_window_size = 256,        # window size of local attention ## local_attn_head=0으로 하면 이 인자 필요 없음   seq_len이 window_size로 나눠지게 하는 게 코드가 깔끔함
    rotary_position_emb = False      # use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding
)
loss = nn.CrossEntropyLoss()
target = torch.empty((BATCH_SIZE, MAX_SEQ_LEN), dtype=torch.long).random_(NUM_TOKENS) # [B, seq_len]
x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, MAX_SEQ_LEN))  # [B, seq_len]
mask = torch.ones_like(x).bool()                             # [B, seq_len]  # NOTE: 임의의 mask를 model에 넣어줄 수 있나? -> global_head에 사용됨. NOTE: 그러나 우리는 마스크 사용하지 않을 것임! 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
mask = mask.to(device)
target = target.to(device)

start =time.time()
output = model(x, mask = mask) # -> [B, seq_len, num_tokens]
output = output.reshape(output.shape[0] * output.shape[1], -1)
target = target.reshape(-1)
L = loss(output, target)
L.backward()
end = time.time()
print("time(s):", end-start)