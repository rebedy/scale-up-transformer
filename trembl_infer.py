import wandb
# wandb.login()
import tracemalloc
import os, sys
import tqdm
import datetime
from time import time, localtime, gmtime, strftime
import math, csv, json, tempfile
from functools import partial
import numpy as np
from pathlib import Path
import glob

import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
#!#  
"""
CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_infer.py --kernel_type generalised --mpi_port 12365
"""
# torch.cuda.empty_cache()
# torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import benchmark
from torch.utils.data import DataLoader, distributed
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.autograd as dist_autograd
from torch import optim
# from apex.fp16_utils import *
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.autograd.profiler import profile, record_function

from torch.cuda.amp import autocast
from torch.autograd import Variable

from utils import *
from helpers import *
from meters import *
from train_helper import *
# from trembl_train import *
from loader_trembl import TREMBLDataset
import trembl_parser as parser_

import os, warnings
warnings.simplefilter("default") # Change the filter in this process
os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses

from performer_pytorch.performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from transformer_pytorch.transformer_pytorch import TransformerLM

start = datetime.datetime.now()
TODAY = str(datetime.date.today().strftime('%Y%m%d'))
NOW = str(start.strftime('_%Hh%Mm'))


parser = argparse.ArgumentParser(parents=[parser_.parser], description="sut on performers")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.ngpus_per_node = torch.cuda.device_count()
gpu = 0

device = torch.device('cuda')
seed = args.seed
random_seed_all(seed)   
cudnn.benchmark = False
cudnn.deterministic = True
torch.set_printoptions(precision=10)


vocab = np.load(args.vocab_path, allow_pickle=True)
pad_idx = vocab["[PAD]"]
args.num_tokens = len(vocab)  # NOTE: text vocab size

## Dataset settings    
print("\nCXR Dataset Loading...")
dirs = glob.glob(args.dataset_dir+"*/")
target_dir = [direc for direc in dirs if str(args.max_seq_len) in direc][0]
args.train_file = glob.glob(target_dir+"train*")[0]
args.val_file = glob.glob(target_dir+"val*")[0]
train_ds = TREMBLDataset(args.train_file, args.max_seq_len, pad_idx)
train_sampler = distributed.DistributedSampler(train_ds, drop_last=True, seed=args.seed) 
train_dl = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, sampler=train_sampler)
val_ds = TREMBLDataset(args.val_file, args.max_seq_len, pad_idx)
val_sampler = distributed.DistributedSampler(val_ds, drop_last=True, seed=args.seed)
val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, sampler=val_sampler)


if args.wandb:
    wandb.init(dir=args.save_dir_base, config = args, project="SUT_PerformersSLiMfin_"+TODAY)
print("\n")
print("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))
print("is cuda available? :", args.cuda)
print("args.world_size ", args.world_size)
print("args.num_workers ", args.num_workers)
print("args.rank ", args.rank)
print("args.local_rank ", args.local_rank)
print("args.batch_size ", args.batch_size)
print("\n")
print("train_ds ", len(train_ds))
print("train_dl ", len(train_dl))
print("val_ds ", len(val_ds))
print("val_dl ", len(val_dl))

## Make DIRs
# exp_name = f"sut_generalised_{TODAY}{NOW}_txt{args.max_seq_len}_h{args.heads}_l{args.depth}_dff{args.d_ff}_d{args.dim}_bz{args.batch_size}"
# if not os.path.exists(Path(args.save_dir_base)):
#     os.makedirs(Path(args.save_dir_base), exist_ok=True)
# save_dir = os.path.join(args.save_dir_base, exp_name+"/")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#!# Settings
if args.infer_model == "transperf":
    args.attn_type = 'transperf'
elif args.infer_model == "trans":
    args.attn_type = 'full'
elif args.infer_model == 'perf':
    args.attn_type = None

if args.kernel_type == 'trans':
    args.generalized_attention = False
    args.no_projection = True
elif args.kernel_type == 'favor':
    args.generalized_attention = False
    args.no_projection = False
elif args.kernel_type == 'generalised':
    args.generalized_attention = True
    args.no_projection = False
    args.clip_grad=False
else:
    raise "The kernel_type argument has to be one of followings; 'trans', 'favor', 'generalised'"

args.causal = 'linear'
args.ff_mult = int(args.d_ff / args.dim)

if args.infer_model == "perf":
    model = PerformerLM(
        max_seq_len = args.max_seq_len,
        num_tokens = args.num_tokens,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        dim_head = args.dim_head,
        local_attn_heads = 0, #(8, 8, 8, 6, 4, 2)*6,
        local_window_size = args.local_window_size,
        causal = args.causal,
        ff_mult = args.ff_mult,
        nb_features = args.nb_features,
        chunk_size = args.chunk_size,
        feature_redraw_interval = args.feature_redraw_interval,
        reversible = args.reversible,
        ff_chunks = args.ff_chunks,
        ff_glu = args.ff_glu,
        emb_dropout = args.emb_dropout,
        ff_dropout = args.ff_dropout,
        attn_dropout = args.attn_dropout,    
        generalized_attention = args.generalized_attention,
        use_scalenorm = args.use_scalenorm,
        use_rezero = args.use_rezero,
        no_projection = args.no_projection,
        tie_embed = args.tie_embed,
        rotary_position_emb = args.rotary_position_emb,
    )

else:
        model = TransformerLM(
        dim = args.dim,
        depth = args.n_layers,
        seq_len = args.max_seq_len,
        num_tokens=args.num_tokens,
        heads = args.n_heads,
        attn_dropout = args.attn_dropout,
        ff_dropout = args.ff_dropout,
        emb_dropout=args.emb_dropout,
        ff_mult = args.ff_mult,
        rotary_emb=args.rotary_emb,
        sandwich_norm=args.sandwich_norm,
        causal=args.causal,
        reversible=args.reversible,
        attn_types=args.attn_type,
        
        generalized_attention = args.generalized_attention,
        no_projection = args.no_projection,
    )
    

# model = PerformerLM(**kargs_performerLM_i2t)
model = AutoregressiveWrapper(model)
model = model.to(memory_format=torch.contiguous_format).cuda(gpu)
model = DistributedDataParallel(model, 
                                device_ids=[gpu],
                                output_device=gpu,
                                find_unused_parameters=True)
scaler = GradScaler(enabled=True)

trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of trainable params: {trainable_parameters}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of TOTAL params: {total_params}")
    
start = time.monotonic()

###!### Training
iter_time, f_time, b_time =[], [], []
for i, batch in enumerate(tqdm.tqdm(train_dl, mininterval=10., desc='training')):
    batch_start = time.monotonic()

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats
    ################################################################################
    seq = batch['seq'].to(gpu).long()#.cuda(args.gpu, non_blocking=True)
    mask = batch['mask'].to(gpu).long()#.cuda(args.gpu, non_blocking=True)
    ################################################################################
    
    b = args.generate_how_many
    model.eval()
    prime = index2base_batch(seq[:b], vocab) #[:-1]
    sample = model.module.generate(seq[:b], args.max_seq_len)#, mask=input_mask)
    output_str = index2base_batch(sample, vocab)
    for itr in range(b):
        print(''.join(prime[itr]), "\n", '*' * 100)
        print(''.join(output_str[itr]), "\n\n")
