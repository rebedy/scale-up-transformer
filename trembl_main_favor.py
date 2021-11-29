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
CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_main_favor.py --mpi_port 12454
CUDA_VISIBLE_DEVICES=5,6 python3 -m torch.distributed.launch --nproc_per_node=2 trembl_main_favor.py --kernel_type favor --mpi_port 12454
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
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
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


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Code is not suited for non distributed mode. Exit.')
        sys.exit(1)
    os.environ['MASTER_PORT'] = str(args.mpi_port)
    # torch.cuda.synchronize()
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)
    torch.backends.cudnn.benchmark


# def main_worker(rank: int, ngpus_per_node:int, args: argparse.Namespace,):
def main_worker(args: argparse.Namespace):
    start = datetime.datetime.now()
    TODAY = str(datetime.date.today().strftime('%Y%m%d'))
    NOW = str(start.strftime('_%Hh%Mm'))

    # #!# DDP
    init_distributed_mode(args)
    gpu = args.gpu
    args.num_workers = 0 #int((8 + args.ngpus_per_node - 1) / args.ngpus_per_node)
    args.local_rank = args.rank * args.ngpus_per_node + gpu
    args.batch_size = int(args.batch_size / args.ngpus_per_node)

    vocab = np.load(args.vocab_path, allow_pickle=True)
    pad_idx = vocab["[PAD]"]
    args.num_tokens = len(vocab)  # NOTE: text vocab size

    ## Dataset settings    
    print("\n TrEMBL Dataset Loading...")
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


    if gpu == 0:
        if args.wandb:
            wandb.init(dir=args.save_dir_base, config = args, project="SUT_PerformersSLiMfinfin_"+TODAY)
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
        exp_name = f"sut_favor_{TODAY}{NOW}_txt{args.max_seq_len}_h{args.heads}_l{args.depth}_dff{args.d_ff}_d{args.dim}_bz{args.batch_size}"
        if not os.path.exists(Path(args.save_dir_base)):
            os.makedirs(Path(args.save_dir_base), exist_ok=True)
        save_dir = os.path.join(args.save_dir_base, exp_name+"/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    
    # if args.kernel_type == 'trans':
    #     args.generalized_attention = False
    #     args.no_projection = True
    # elif args.kernel_type == 'favor':
    #     args.generalized_attention = False
    #     args.no_projection = False
    # elif args.kernel_type == 'generalised':
    #     args.generalized_attention = True
    #     args.no_projection = False
    #     args.clip_grad=False
    # else:
    #     raise "The kernel_type argument has to be one of followings; 'trans', 'favor', 'generalised'"
        
    args.kernel_type = 'favor'
    args.generalized_attention = False
    args.no_projection = False 
    args.causal = 'linear'
    args.clip_grad=False
    args.ff_mult = int(args.d_ff / args.dim)
    
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
    
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay)

    dist.barrier()

    print(model)
    exit()
    
    # Scale learning rate based on global batch size 1*10
    # args.lr = args.lr*float(args.batch_size*args.world_size)/256.
            
    if os.path.exists(args.ckpt_dir):
        print("Loading Checkpoint... '{}'".format(args.ckpt_dir))
        checkpoint = torch.load(args.ckpt_dir, map_location = lambda storage, loc: storage.cuda(gpu))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    
    if gpu == 0 and args.wandb:
        wandb.watch(model)
     
        
        
     
    #!!!!!!!!!!!!!!!!!!!!!!!!!!# Benchmark #!!!!!!!!!!!!!!!!!!!!!!!!!!#
    
    print("\nStart Training.. Favor", args.kernel_type)
    ep_secs = []
    best_loss = 1.0
    tot_start = time.monotonic()
    
    for epoch in range(args.start_epoch, args.epochs):
        start = time.monotonic()
        
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        ###!### Training
        iter_time, f_time, b_time =[], [], []
        for i, batch in tqdm.tqdm(enumerate(train_dl), mininterval=10., leave=True, 
                                  desc=f'Epoch-{epoch} Iterator', total=len(train_dl),bar_format='{l_bar}{bar:10}{r_bar}'):
            batch_start = time.monotonic()
            adjust_learning_rate(args.lr, optimizer, epoch, i, len(train_dl))

            model.train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats
            ################################################################################
            seq = batch['seq'].to(gpu).long()#.cuda(args.gpu, non_blocking=True)
            mask = batch['mask'].to(gpu).long()#.cuda(args.gpu, non_blocking=True)
            ################################################################################
        
            # for __ in range(args.gradient_accum):
            #!#
            with autocast():
                forward_start = time.monotonic()
                
                if args.profiler:
                    with torch.autograd.profiler.profile(profile_memory=True) as prof: #with_stack=True, 
                        logits, loss = model(seq)
                else:
                    logits, loss = model(seq)
                    
            #!#
            dist.barrier()
                    
            if gpu == 0:
                f_time.append(np.log2(time.monotonic() - forward_start))
                forward_time = np.log2(time.monotonic() - forward_start)
                backward = time.monotonic()
                           
            #!#
            scaler.scale(loss).backward()
            #!#
            
            if args.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                    
            if gpu == 0:
                batch_time = time.monotonic()-batch_start
                backward_time = np.log2(time.monotonic() - backward)
                ppl = math.exp(loss)
                train_bpd = loss.item() / np.log(2)
                gb_in_use = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
                logits = logits.transpose(1, 2)
                correct = (seq[:, 1:] == torch.argmax(logits, 1))
                hit_sum = correct.float().sum()
                    
            if gpu == 0 and args.profile:
                iter_time.append(time.monotonic()-batch_start)
                b_time.append(np.log2(time.monotonic() - backward))
                
            if i==30 and gpu == 0 and args.profile:
                if args.profiler==True:
                    print("\ncpu_time_total")
                    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=5))
                    print("\nself_cuda_memory_usage")
                    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=5))
                    # print(prof.key_averages().table(sort_by="cpu_time_total"))
                    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                    
                print("\n")
                print("gb_in_use torch.cuda.max_memory_allocated", round(gb_in_use, 3))
                print("Model Forward Timing (log_2(T)(sec)", round(np.mean(f_time), 3))
                print("Model Backward Timing (log_2(T)(sec)", round(np.mean(b_time), 3))            
                print("\n\n")
                print("iter time : ",  round(np.mean(iter_time), 3), "\n")
                print("\n")
                exit()
                

            if gpu == 0 and args.wandb:
                wandb.log({
                "train avg speed per batch process": round(batch_time, 3),
                "train bits per dimension": train_bpd,
                "train gigabites in use": gb_in_use,
                "train hit_sum": hit_sum,
                "train loss": loss,
                "train ppl": ppl,
                "Model Forward Timing (log_2(T)(sec)": round(forward_time, 3),
                "Model Backward Timing (log_2(T)(sec)": round(backward_time, 3),
                })
                
                
            ###!### Validation
            if int(i+1) % args.validate_every == 0 and i != 0:
                model.eval()
                _val_avg_loss, _val_avg_ppl = [], []
                with torch.no_grad():
                    for j, batch in tqdm.tqdm(enumerate(val_dl), mininterval=10., leave=True, 
                                              desc=f'validation', total=len(val_dl), bar_format='{l_bar}{bar:10}{r_bar}'):
                        seq = batch['seq'].to(gpu).long()#.cuda(args.gpu, non_blocking=True)
                        val_logits, val_loss = model(seq)
                        val_loss = val_loss.item()
                        
                        if gpu == 0:
                            _val_avg_loss.append(val_loss)
                            val_ppl = math.exp(val_loss)
                            _val_avg_ppl.append(val_ppl)
                            val_logits = val_logits.transpose(1, 2)
                            correct = (seq[:, 1:] == torch.argmax(val_logits, 1))
                            val_hit_sum = correct.float().sum()
                        
                        if gpu == 0 and args.wandb:
                            wandb.log({
                            "valid hit_sum": val_hit_sum,
                            "valid loss": val_loss,
                            "valid ppl": val_ppl,
                            })
                
                if gpu == 0:
                    val_avg_loss = np.array(_val_avg_loss).mean()
                    val_avg_ppl = np.array(_val_avg_ppl).mean()
                    print(f'\nEpoch {epoch}: validation loss: {val_avg_loss} | validation ppl: {val_avg_ppl}\n')
                  
                
            
            if int(i+1) % args.generate_every == 0 and i != 0 and gpu ==0:
                b = args.generate_how_many
                model.eval()
                prime = index2base_batch(seq[:b], vocab) #[:-1]
                sample = model.module.generate(seq[:b], args.max_seq_len)#, mask=input_mask)
                output_str = index2base_batch(sample, vocab)
                for itr in range(b):
                    print(''.join(prime[itr]), "\n", '*' * 100)
                    print(''.join(output_str[itr]), "\n\n")
        
        # save model & log
        if gpu == 0:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch,
                'model': "transformer",
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
                }, 
                            is_best, 
                            filename=os.path.join(save_dir, f"epoch={epoch:06}.ckpt"))
            print("\n", epoch, " epoch took for", time.monotonic()-start)
            
        if not math.isfinite(loss):
            print(f'Evaluation loss is {loss}, stopping training!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit(1)

    dist.destroy_process_group()
    print("\nThe whole ", epoch, " epoch took for", time.monotonic()-tot_start, "\n", args.kernel_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[parser_.parser], description="sut on performers")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.ngpus_per_node = torch.cuda.device_count()
    
    device = torch.device('cuda')
    seed = args.seed
    random_seed_all(seed)   
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_printoptions(precision=10)

    main_worker(args)
