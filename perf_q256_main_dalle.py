import wandb
# wandb.login()

import os, sys
from time import time, localtime, gmtime, strftime
import tqdm, datetime, timeit
import math, csv, json, tempfile
from functools import partial
import numpy as np
from pathlib import Path

import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  #!# Donot erase or commentize
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
# torch.cuda.empty_cache()
# torch.set_printoptions(profile="full")
#!#  
""" 
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 perf_q256_main_dalle.py
"""
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

from torch.cuda.amp import autocast
from torch.autograd.profiler import profile, record_function
from torch.autograd import Variable

from utils import *
from meters import *
from helpers import *
from train_helper import *

from loader import CXRDataset
from transformer_dalle.transformer_i2t import *
import perf_q256_parser_dalle as parser_

MPI_PORT = 12367

## helpers

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

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
    os.environ['MASTER_PORT'] = str(MPI_PORT)
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
    

def train_one_epoch(gpu, model, train_loader, scaler, optimizer, epoch, args):
    epoch_time = AverageMeter('Time', ':.3f')
    batch_time = AverageMeter('Batch Time', ':.3f')
    losses = AverageMeter('Loss', ':.4e')
    perplexities = AverageMeter('PPL', ':.4e')
    if gpu == 0:
        progress = ProgressMeter( len(train_loader), [batch_time, losses], prefix='Training: ')
        ep_start = time.monotonic()
        end = time.monotonic()
        
    ## switch to train mode
    model.train()

    cnt = 0
    for i, batch in enumerate(tqdm.tqdm(train_loader)):
        images = batch['images'].to(gpu)#.cuda(args.gpu, non_blocking=True)
        texts = batch['texts'].to(gpu)#.cuda(args.gpu, non_blocking=True)
        
        adjust_learning_rate(args.lr, optimizer, epoch, i, len(train_loader))
        ## compute gradient and do the Optimizer step
        optimizer.zero_grad()
        with autocast():
            if args.profile:
                with torch.autograd.profiler.profile( use_cuda=True, profile_memory=True,record_shapes=True,) as prof:
                    logits =  model(images, texts)
            else:
                logits = model(images, texts) # [16, 512, 14526])
            # logit = logits[:, args.condition_len:-1].reshape(-1, logits.size(-1))
            # target = texts.reshape(-1)
            logit = logits[:, args.condition_len:-1].reshape(-1, logits.size(-1))
            target = texts[:, 1:].reshape(-1)
            loss = F.cross_entropy(logit, target, ignore_index=args.pad_token_idx).cuda(gpu)
           
            if i>3 and not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
                
        #!# Loss scale and step and update
        ## Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        ## Backward passes under autocast are not recommended.
        ## Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()
        
        ## scaler.step() first unscales the gradients of the optimizer's assigned params.
        ## If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        ## otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        ## Updates the scale for next iteration.
        scaler.update()           
        ## gradient clipping
            # if args.clip_grad:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
                
        ## Every print_freq iterations, check the loss, accuracy, and speed.
        ## For best performance, it doesn't make sense to print these metrics every
        ## iteration, since they incur an allreduce and some host<->device syncs.
        torch.cuda.synchronize(gpu)
        
        ##!## Measurements
        ### TODO | DY: add bpd and gd??? get it from SLiM performer
        ### Average loss and accuracy across processes for logging
        reduced_loss = reduce_tensor(args.world_size, loss.data)
        ### to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), images.size(0))
            
        ppl = torch.exp(reduced_loss)
        perplexities.update(to_python_float(ppl), images.size(0))
            
        
        if gpu == 0:
            batch_time.update((time.monotonic() - end))
            end = time.monotonic()
        if gpu == 0 and i % args.print_freq==0:
            progress.display(i)
        if gpu == 0 and args.profile==True and cnt<2:
            print(prof.key_averages().table(sort_by="cpu_time_total"))
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
        cnt+=1
        
    if gpu == 0:
        epoch_time.update((time.monotonic() - ep_start))
        print('Epoch: [{0}][{1}/{2}]\t'
                'Time {time_avg} (tot. {time_val})\t'
                'Loss {loss.avg:.10f} (fin. {loss.val:.4f})\t'
                'PPL {ppl.avg:.10f} (fin. {ppl.val:.4f})\t'
                'batch_time {bt_time_avg} ({bt_time_val})'.format(
                    epoch, i+1, len(train_loader),
                    strftime("%H:%M:%S", gmtime(args.world_size*args.batch_size/epoch_time.val)), #speed
                    strftime("%H:%M:%S", gmtime(args.world_size*args.batch_size/epoch_time.avg)),
                    loss=losses,
                    ppl=perplexities,
                    epoch_time=epoch_time,
                    time_val = strftime("%H:%M:%S", gmtime(epoch_time.val)), #time
                    time_avg = strftime("%H:%M:%S", gmtime(epoch_time.avg)),
                    batch_time=batch_time,
                    bt_time_val = strftime("%H:%M:%S", gmtime(batch_time.val)), #time
                    bt_time_avg = strftime("%H:%M:%S", gmtime(batch_time.avg)),
                    ))
        if args.wandb:
            wandb.log({
                "train tot time per epoch process": epoch_time.avg,
                "train avg speed per epoch process": args.world_size*len(train_loader)/epoch_time.avg,
                "train avg time per batch process": batch_time.avg,
                "train avg speed per batch process": args.world_size*args.batch_size/batch_time.avg,
                "train avg loss per epoch process": losses.avg,
                "train avg ppl per epoch process": perplexities.avg,
            }, step=epoch-1)
    
    

@torch.no_grad()
def evaluate(gpu, model, val_loader, epoch, args):
    epoch_time = AverageMeter('Time', ':.3f')
    batch_time = AverageMeter('Batch Time', ':.3f')
    losses = AverageMeter('Loss', ':.4e')
    perplexities = AverageMeter('PPL', ':.4e')
    
    if gpu == 0:
        progress = ProgressMeter( len(val_loader),
                                 [batch_time, losses], prefix='Valid: ')
        ep_start = time.monotonic()
        end = time.monotonic()
        
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            images = batch['images'].to(gpu)#.cuda(args.gpu, non_blocking=True)
            texts = batch['texts'].to(gpu)#.cuda(args.gpu, non_blocking=True)
         
            ## compute output
            logits =  model(images, texts)
            logit = logits[:, args.condition_len:-1].reshape(-1, logits.size(-1))
            target = texts[:, 1:].reshape(-1)
            loss = F.cross_entropy(logit, target, ignore_index=args.pad_token_idx).cuda(gpu)

            torch.cuda.synchronize(gpu)
            
            ## measure accuracy and record loss
            ## TODO | DY: add bpd and gd??? get it from SLiM performer
            
            #!# Accumulates scaled gradients.
            reduced_loss = reduce_tensor(args.world_size, loss.data)
            ### to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))
            
            ppl = torch.exp(reduced_loss)
            perplexities.update(to_python_float(ppl), images.size(0))
            
            if gpu == 0:
                ## measure elapsed time
                batch_time.update(time.monotonic() - end)
                end = time.monotonic()
            if gpu == 0 and i+1 % args.print_freq == 0:
                progress.display(i)
        if gpu == 0:
            epoch_time.update((time.monotonic() - ep_start))
            # TODO:  Change timings to mirror train().
            print('Test: [{0}/{1}]\t'
                    'Time {time_avg} (tot. {time_val})\t'
                    'Loss {loss.avg:.10f} (fin. {loss.val:.4f})\t'
                    'PPL {ppl.avg:.10f} (fin. {ppl.val:.4f})\t'
                    'batch_time {bt_time_avg} ({bt_time_val})'.format(
                        i+1, len(val_loader),
                        strftime("%H:%M:%S", gmtime(args.world_size*args.batch_size/epoch_time.val)), #speed
                        strftime("%H:%M:%S", gmtime(args.world_size*args.batch_size/epoch_time.avg)),
                        loss=losses,
                        ppl=perplexities,
                        epoch_time=epoch_time,
                        time_val = strftime("%H:%M:%S", gmtime(epoch_time.val)), #time
                        time_avg = strftime("%H:%M:%S", gmtime(epoch_time.avg)),
                        batch_time=batch_time,
                        bt_time_val = strftime("%H:%M:%S", gmtime(batch_time.val)), #time
                        bt_time_avg = strftime("%H:%M:%S", gmtime(batch_time.avg)), 
                        ))
            if args.wandb:
                wandb.log({
                    'Valid:' : i,
                    "val tot time per epoch process": epoch_time.avg,
                    "val avg speed per epoch process": args.world_size*len(val_loader)/epoch_time.avg,
                    "val avg time per batch process": batch_time.avg,
                    "val avg speed per batch process": args.world_size*args.batch_size/batch_time.avg,
                    "val avg loss per epoch process": losses.avg,
                    "val avg ppl per epoch process": perplexities.avg,
                }, step=epoch-1)
    return losses.avg


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
    
    ## Dataset settings    
    print("\nCXR Dataset Loading...")    
    dataset_args={
        "img_root_dir" : args.img_root_dir,
        "text_root_dir" : args.text_root_dir,
        "vqgan_model_path" : args.vqgan_model_path,
        "vqgan_config_path" : args.vqgan_config_path,
        "codebook_indices_path" : args.codebook_indices_path,
        "vocab_file" : args.vocab_file,
        "merge_file" : args.merge_file,
        "target_count" : args.target_count,
        "max_img_num" : args.max_img_num,
        "under_sample" : args.under_sample_type,
        "max_text_len" : args.max_text_len,
    }
    train_ds = CXRDataset(args.train_meta_file, **dataset_args)
    train_sampler = distributed.DistributedSampler(train_ds, drop_last=True, seed=args.seed) 
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, sampler=train_sampler)
    val_ds = CXRDataset(args.val_meta_file, **dataset_args)
    val_sampler = distributed.DistributedSampler(val_ds, drop_last=True, seed=args.seed)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, sampler=val_sampler)

    if gpu == 0:
        if args.wandb:
            wandb.init(dir=args.save_dir_base, config = args, project="SUT_Performers_"+TODAY)
        # train_ds  81271
        # train_dl  6773
        # val_ds  678
        # val_dl  57
        # test_ds  1305
        # test_dl  109
        print("\n")
        print(" >>> Performers Quantized 256 version!!!!!")
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
        exp_name = f"sut_q256_{TODAY}{NOW}_totim{args.target_count}_im{args.max_img_num}_txt{args.max_text_len}_{args.under_sample_type}_bz{args.batch_size}_ep{args.epochs}_lr{args.lr}_{args.activation}"
        if not os.path.exists(Path(args.save_dir_base)):
            os.makedirs(Path(args.save_dir_base), exist_ok=True)
        save_dir = os.path.join(args.save_dir_base, exp_name+"/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    tokenizer_ = train_ds.tokenizer
    args.pad_token_idx = tokenizer_.token_to_id("[PAD]")
    
    ## add args
    args.num_tokens = train_ds.text_vocab_size  # NOTE: text vocab size
    args.num_img_tokens = train_ds.img_vocab_size + train_ds.max_img_num   # NOTE: img vocab size + num img pad
    args.max_seq_len = train_ds.img_len * train_ds.max_img_num + train_ds.max_text_len
    args.max_img_num = train_ds.max_img_num
    args.max_text_len = train_ds.max_text_len
    args.condition_len = train_ds.img_len * train_ds.max_img_num
    print("condition_len is : ", args.condition_len)
    args.img_fmap_size = int(train_ds.img_fmap_size)  #16

    kargs_transformer_i2t = {
        'dim':args.dim,
        'depth':args.depth,
        'num_tokens':args.num_tokens,
        'num_img_tokens':args.num_img_tokens,
        'max_seq_len':args.max_seq_len,
        'max_img_num':args.max_img_num,
        'padding_idx' : args.pad_token_idx,
        'condition_len': args.condition_len,
        'image_fmap_size':args.img_fmap_size,
        
        'heads':args.heads,
        'dim_head':args.dim_head,
        'reversible':args.reversible,
        'causal':args.causal,
        
        'attn_dropout':args.attn_dropout,
        'ff_dropout':args.ff_dropout,
        'attn_types':args.attn_types,
        'image_fmap_size':args.img_fmap_size,
        'sparse_attn':args.sparse_attn,
        'stable':args.stable_softmax,
        'shift_tokens':args.shift_tokens,
        'rotary_emb':args.rotary_emb,
        'emb_dropout':args.emb_dropout,
        'tie_embed':args.tie_embed,
        }
    
    model = TransformerLM_i2t(**kargs_transformer_i2t)
    model = model.to(memory_format=torch.contiguous_format).cuda(gpu)
    model = DistributedDataParallel(model, 
                                    device_ids=[gpu],
                                    output_device=gpu,
                                    find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of params: {n_parameters}")
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8,15,18], gamma=0.5)

    dist.barrier()
    
    # Scale learning rate based on global batch size 1*10-5
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

    ## Benchmark
    print("\nStart Training..")
    ep_secs = []
    best_loss = 0
    tot_start = datetime.datetime.now()
    
    for epoch in range(args.start_epoch, args.epochs+1):
        start = datetime.datetime.now()
        
        # print(f"Epoch: {epoch}")
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # scheduler.step()
        
        train_one_epoch(gpu, model, train_dl, scaler, optimizer, epoch, args)
        loss = evaluate(gpu, model, val_dl, epoch, args)
                
        # # Without the join() API, the below synchronization will hang blocking for rank 1's allreduce to complete.
        # torch.cuda.synchronize()
        
        # remember best prec@1 and save checkpoint
        if gpu == 0:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': "transformer",
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
                }, 
                            is_best, 
                            filename=os.path.join(save_dir, f"epoch={epoch:06}.ckpt"))
            print("\n", epoch, " epoch took for", datetime.datetime.now()-start)
        
        if not math.isfinite(loss):
                print(f'Evaluation loss is {loss}, stopping training!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                sys.exit(1)
            
    dist.destroy_process_group()
    print("\nThe whole ", epoch, " epoch took for", datetime.datetime.now()-tot_start, "\n")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[parser_.parser], description="sut on transformer(dalle)")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.ngpus_per_node = torch.cuda.device_count()
    
    device = torch.device('cuda')
    seed = args.seed
    random_seed_all(seed)   
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_printoptions(precision=10)
    
    args.cuda = torch.cuda.is_available()
    args.ngpus_per_node = torch.cuda.device_count()
    
    main_worker(args)
    # mp.spawn(main_worker, args=(args), nprocs=args.ngpus_per_node,)
    
    
    ### TODO ################################################################################
    """
    MASTER_PORT: 0-순위의 프로세스를 호스트할 기기의 비어있는 포트 번호(free port)
    MASTER_ADDR: 0-순위의 프로세스를 호스트할 기기의 IP 주소
    WORLD_SIZE: 전체 프로세스 수 - 마스터가 얼마나 많은 워커들을 기다릴지 알 수 있음
    RANK: 각 프로세스의 우선순위 - 워커의 마스터 여부를 확인할 수 있음

    "Pure FP32" training:
    $ python main_amp.py -a resnet5 0 --b 128 --workers 4 --opt-level O0 ./

    Pure FP16" training:
    $ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O3 ./
    
    --opt-level O1 (Official Mixed Precision recipe, recommended for typical use)
    
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=7778 main_trans.py
    
    python -m torch.distributed.launch --nproc_per_node=2 --nnode=2 --node_rank=0 --master_addr='10.0.3.29' --master_port=9901 main_trans.py
    python -m torch.distributed.launch --nproc_per_node=2 --master_addr='10.0.3.29' --master_port=9901 main_trans.py
    python -m torch.distributed.launch --nproc_per_node=1 main_dalle.py
    
    bs 1 = 3184 MB
    bs 26 = 23388 MB
    
    bs 128
    RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB 
    (GPU 0; 23.70 GiB total capacity; 21.82 GiB already allocated; 585.69 MiB free; 21.91 GiB reserved in total by PyTorch)
    
    bs 36
    RuntimeError: CUDA out of memory. Tried to allocate 432.00 MiB 
    (GPU 0; 23.70 GiB total capacity; 22.07 GiB already allocated; 403.69 MiB free; 22.09 GiB reserved in total by PyTorch)
   
    bs 24
    RuntimeError: CUDA out of memory. Tried to allocate 144.00 MiB 
    (GPU 3; 23.70 GiB total capacity; 22.21 GiB already allocated; 135.69 MiB free; 22.35 GiB reserved in total by PyTorch)
    
    bs 18
    RuntimeError: CUDA out of memory. Tried to allocate 510.00 MiB 
    (GPU 0; 23.70 GiB total capacity; 22.28 GiB already allocated; 121.69 MiB free; 22.36 GiB reserved in total by PyTorch)
    
    <Single node, multi GPU, DDP>
    
    python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=4 --master_port=7778 main_trans.py
    python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=2 --master_port=78789 main_trans.py
        
        
        


    
    RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
    This error indicates that your module has parameters that were not used in producing loss. 
    You can enable unused parameter detection by 
    (1) passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`; 
    (2) making sure all `forward` function outputs participate in calculating loss. 
    If you already have done the above two steps, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. 
    Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
    
"""