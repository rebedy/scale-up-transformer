import argparse
from helpers import *


parser = argparse.ArgumentParser(add_help=False)

## dataset args
# #!#
parser.add_argument('--train_mini_meta_file', default=['metadata/mdvl_Train_mini.jsonl', 'metadata/mdvl_mimiccxr_train_mini.csv'], type=str)
parser.add_argument('--train_meta_file', default=['metadata/mdvl_Train.jsonl', 'metadata/mdvl_mimiccxr_train.csv'], type=list) 
parser.add_argument('--val_meta_file', default=['metadata/mdvl_Valid.jsonl','metadata/mdvl_mimiccxr_valid.csv'], type=list)
parser.add_argument('--test_meta_file', default=['metadata/mdvl_Test.jsonl','metadata/mdvl_mimiccxr_test.csv'], type=list)
parser.add_argument('--img_root_dir', default='/home/data_storage/dylee_135/mimic-mdvl-preprocessed/re_512_3ch', type=str)
parser.add_argument('--text_root_dir', default='/home/data_storage/dylee_135/mimic-mdvl-preprocessed/preprocessed_reports_mdvl', type=str)
parser.add_argument('--vqgan_model_path', default='/home/data_storage/dylee_135/mimic-vqgan/mimiccxr_vqgan1024/checkpoints/last.ckpt', type=str)
parser.add_argument('--vqgan_config_path', default='/home/data_storage/dylee_135/mimic-vqgan/mimiccxr_vqgan1024/configs/2021-07-05T10-23-24-project.yaml', type=str)
parser.add_argument('--codebook_indices_path', default="/home/data_storage/dylee_135/mimic-vqgan/codebook_indices/mimiccxr_vqgan1024_codebook_indices.pickle", type=str) #
parser.add_argument('--vocab_file', default='BBPE_tokenizer/vocab.json', type=str)
parser.add_argument('--merge_file', default='BBPE_tokenizer/merges.txt', type=str)


#!#
parser.add_argument('--target_count', default=1, type=int) # 이미지 몇장을 가진 study들을 가지고 올지.
parser.add_argument('--max_img_num', default=1, type=int) # 내가 보여주고싶은 최대 이미지수
parser.add_argument('--max_text_len', default=256, type=int) # report 최대 길이
parser.add_argument('--img_resol', default=256, type=int) # image albumentation preprocessing
## when target_count > max_img_num
parser.add_argument('--under_sample_type', default='fixed', choices=["fixed","random"], type=str)
    
## model args
parser.add_argument('--dim', default=768, type=int, help='Model dimension. The dimension must be divisible by number of heads.')
parser.add_argument('--depth', default=12, type=int, help='Model depth. Layers.')
parser.add_argument('--heads', default=12, type=int, help='Model number of heads. Attention heads.')
parser.add_argument('--dim_head', default=64, type=int, help='Model head dimension. dim of head. inner_dim = dim_head * heads') # projection matrix에 의해 head별 차원수: dim_head -> nb_fetures
parser.add_argument('--emb_dropout', default=0.1, type=float, help='embedding dropout')
parser.add_argument('--ff_dropout', default=0.1, type=float, help='Feed forward dropout.')
parser.add_argument('--attn_dropout', default=0.1, type=float, help='Post-attn dropout.')
parser.add_argument('--causal', default=True, type=str2bool, help='auto-regressive or not')
parser.add_argument('--reversible', default=False, type=str2bool, help='reversible layers, from Reformer paper. Works only when sharded_ddp=True')
parser.add_argument('--tie_embed', default=False, type=str2bool, help='multiply final embeddings with token weights for logits')
parser.add_argument('--rotary_emb', default=False, type=str2bool, 
                    help='Use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. \
                        hould always be turned on unless if you want to go back to old absolute positional encoding')
parser.add_argument('--attn_types', default='conditioned_i2t', type=str, choices=['conditioned_i2t', 'favor+', 'full','sparse','axial_row','axial_col','conv_like','mlp',],
                    help='comma separated list of attention types. \
                        attention type can be: full or sparse or axial_row or axial_col or conv_like or conditioned_i2t.')
parser.add_argument('--shift_tokens', help = 'Use the shift tokens feature', action = 'store_true')
parser.add_argument('--stable_softmax', dest='stable_softmax', action='store_true',
                help='Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.')
parser.add_argument('--nb_features', default=64, type=int, help='number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head.')
parser.add_argument('--generalized_attention', default=False, type=str2bool, help='defaults to softmax approximation, but can be set to True for generalized attention')


# training args
                        #!#
parser.add_argument('--batch_size', default=60, type=int)
parser.add_argument('--ckpt_dir', default="sut_c256_20211014_15h49m_totim1_im1_txt256_fixed_bz30_ep50_lr1e-05_relu/epoch=000050.ckpt",type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--clip_grad', default=False, type=str2bool)
parser.add_argument('--max_grad_norm', default=5., type=float)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--activation', default="relu", type=str)

##! DDP 
parser.add_argument('--wandb', default=True, type=bool, help='Whether to use wandb log or not.') 
## FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied automatically by torch.distributed.launch.
parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
# parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
## you need this argument in your scripts for DDP to work
parser.add_argument('--local_rank', type=int, default=0, help='Local process rank.') 
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
parser.add_argument('--profile', default=True, type=bool, help='Only run 10 iterations for profiling.')
parser.add_argument('--fp16_allreduce', action='store_true', default=True,help='use fp16 compression during allreduce')
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
parser.add_argument('--loss_scale', type=str, default=None)
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--save_dir_base', default="/home/edlab/dylee/scaleup_transformer/transformer_c256/" ,type=str)

if __name__ == '__main__':
    args = parser.parse_args()
