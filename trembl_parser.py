"""
CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_trans.py --mpi_port 12476

CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_main_favor.py --kernel_type favor --mpi_port 12454
CUDA_VISIBLE_DEVICES=5 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_main_generalised.py --kernel_type generalised --mpi_port 12365

CUDA_VISIBLE_DEVICES=6 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_transperf_favor.py --kernel_type favor --mpi_port 12512
CUDA_VISIBLE_DEVICES=7 python3 -m torch.distributed.launch --nproc_per_node=1 trembl_transperf_generalised.py --kernel_type generalised --mpi_port 16523
"""
import argparse
from helpers import *

parser = argparse.ArgumentParser(add_help=False)

## dataset args
parser.add_argument('--dataset_dir', default="/home/data_storage/dylee_135/trembl/", type=str)
parser.add_argument('--vocab_path', default="metadata/trembl_vocab.pkl", type=str)
                        #!#
parser.add_argument('--max_seq_len', default=256, type=int) # report 최대 길이


## Model args
                        #!#
parser.add_argument('--dim', default=512, type=int, help='model dimension. dimension must be divisible by number of heads.')
parser.add_argument('--depth', default=6, type=int, help='layers')
parser.add_argument('--heads', default=8, type=int, help='heads')
parser.add_argument('--dim_head', default=64, type=int, help='dim of head. inner_dim = dim_head * heads. d_k')
parser.add_argument('--d_ff', default=2048, type=int, help='dim * mult.  In feed forward layer, self.w1 = nn.Linear(dim, dim * mult) and self.w2 = nn.Linear(dim * mult, dim)')
parser.add_argument('--nb_features', default=256, type=int, help='number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head.')
                        #!#
parser.add_argument('--attn_types', default="full", choices=["full", "perf"], type=str, 
                    help='comma separated list of attention types. Attention type can be: full or sparse or axial_row or axial_col or conv_like or conditioned_i2t.')
parser.add_argument('--kernel_type', default="favor", choices=["favor", "generalised", "no_projection"], type=str, help='')
parser.add_argument('--causal', default="linear", choices=[None, "conditioned", "linear"], type=str, help='auto-regressive or not')
                        #!#
                        
parser.add_argument('--chunk_size', default=None,  help='chunk size. if None, same as seqence length.')
parser.add_argument('--reversible', default=False, type=str2bool, help='reversible layers, from Reformer paper. Works only when sharded_ddp=True')
parser.add_argument('--shift_tokens', help = 'Use the shift tokens feature', action = 'store_true')

##! Performer 
parser.add_argument('--local_attn_heads', default=0, type=int, help='if n heads are local attention, heads-n others are global performers.')
parser.add_argument('--local_window_size', default=256, type=int, help='window size of local attention')
parser.add_argument('--feature_redraw_interval', default=1000, type=int, help='how frequently to redraw the projection matrix, the more frequent, the slower the training')
parser.add_argument('--ff_chunks', default=10, type=int, help='chunk feedforward layer, from Reformer paper')
parser.add_argument('--ff_glu', default=True, type=str2bool, help='use GLU variant for feedforward')
parser.add_argument('--emb_dropout', default=0.1, type=float, help='embedding dropout')
parser.add_argument('--ff_dropout', default=0.1, type=float, help='feedforward dropout')
parser.add_argument('--attn_dropout', default=0.1, type=float, help='post-attn dropout')
parser.add_argument('--use_scalenorm', default=False, type=str2bool, help='use scale norm, from Transformers without Tears paper')
parser.add_argument('--use_rezero', default=False, type=str2bool, help='use rezero, from Rezero is all you need paper')  # scalenorm, rezero, layernorm 중 한가지만 사용 가능.
parser.add_argument('--tie_embed', default=False, type=str2bool, help='multiply final embeddings with token weights for logits')
parser.add_argument('--rotary_position_emb', default=False, type=str2bool, help='use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding')

##! Transformer 
parser.add_argument('--stable_softmax', default=True, type=str2bool,
            help='Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.')
parser.add_argument('--sandwich_norm', default=False, type=str2bool, help='use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding')
parser.add_argument('--rotary_emb', default=False, type=str2bool, 
                    help='Use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. \
                        hould always be turned on unless if you want to go back to old absolute positional encoding')

# training args
                        #!#
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--ckpt_dir', default="",type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--clip_grad', default=False, type=str2bool)
parser.add_argument('--max_grad_norm', default=5., type=float)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--activation', default="relu", type=str)

##! LOG 
parser.add_argument('--wandb', default=True, type=str2bool, help='Whether to use wandb log or not.') 
parser.add_argument('--profiler', default=False, type=bool, help='Only run 10 iterations for profiling.')
parser.add_argument('--profile', default=True, type=bool, help='Only run 10 iterations for profiling.')
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--validate_every', '-ve', default=2000, type=int, help='validate frequency (default: 100)')
parser.add_argument('--generate_every', '-ge', default=5000, type=int, help='generate frequency (default: 500)')
parser.add_argument('--generate_how_many', '-ghm', default=2, type=int, help='generate frequency (default: 500)')
parser.add_argument('--save_dir_base', default="/home/edlab/dylee/scaleup_transformer/TrEMBL_fin/" ,type=str)

# inference args
parser.add_argument('--infer_ckpt', default="/home/edlab/dylee/scaleup_transformer/TrEMBL_fin/" ,type=str)

##! DDP 
## FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied automatically by torch.distributed.launch.
parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
# parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
## you need this argument in your scripts for DDP to work
parser.add_argument('--local_rank', type=int, default=0, help='Local process rank.') 
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
parser.add_argument('--fp16_allreduce', action='store_true', default=True,help='use fp16 compression during allreduce')
parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
parser.add_argument('--loss_scale', type=str, default=None)
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--mpi_port', type=int, default=12345, help='') 
    
if __name__ == '__main__':
    args = parser.parse_args()