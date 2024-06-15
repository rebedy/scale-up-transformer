import os
import argparse
import datetime
from functools import partial
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
import torch.nn as nn

from t2i_plmodel import PerformerLightning_t2i, TransformerLightning_t2i
from datamodule import CXRDataModule
from loader import CXRDataset
from helpers import str2bool
from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


"""

CUDA_VISIBLE_DEVICES=2,3 python3 t2i_main.py --transformer --FAVOR --generalized_attention --attn_type conditioned_cuda --target_count 2 --max_img_num 2
CUDA_VISIBLE_DEVICES=6,7 python3 t2i_main.py --transformer --FAVOR --generalized_attention --attn_type conditioned_cuda --target_count 2 --max_img_num 2

"""
if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument(
        '--train_meta_file', default='metadata/mimiccxr_train_sub_final.csv', type=str)
    parser.add_argument(
        '--val_meta_file', default='metadata/mimiccxr_validate_sub_final.csv', type=str)
    parser.add_argument(
        '--test_meta_file', default='metadata/mimiccxr_test_sub_final.csv', type=str)
    parser.add_argument(
        '--img_root_dir', default='/home/edlab/dylee/mimic/sut_image_preprocessing/', type=str)
    parser.add_argument(
        '--text_root_dir', default='/home/edlab/dylee/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/preprocessed_reports', type=str)

    parser.add_argument('--vqgan', default=512, type=int,
                        help='must be less than or equal to target_count')
    parser.add_argument('--max_img_num', default=2, type=int,
                        help='must be less than or equal to target_count')
    parser.add_argument('--target_count', default=2, type=int)
    parser.add_argument('--max_text_len', default=256, type=int)
    parser.add_argument(
        '--vocab_file', default='BBPE_tokenizer/vocab.json', type=str)
    parser.add_argument(
        '--merge_file', default='BBPE_tokenizer/merges.txt', type=str)
    parser.add_argument(
        '--target_view', default=['AP', 'PA', 'LATERAL', 'LL'], nargs='+', type=str)
    parser.add_argument('--under_sample', default='fixed',
                        choices=['fixed', 'random'], type=str)

    # training args
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--reload_ckpt_dir', default=None, type=str)
    # parser.add_argument('--reload_ckpt_dir', default='/home/edlab/dylee/scaleup_transformer/t2i_Performers//t2i_conditioned_causal_2of2_20220329_18h03m_d256_l4_h4_conditioned_cuda_res512_lr_0.001_trans_gen/last.ckpt', type=str)
    parser.add_argument('--seed', default=42, type=int)

    # ! #
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--beam_size', default=1, type=int)
    parser.add_argument('--accumulate_grad_batches', default=16, type=float)
    parser.add_argument('--n_gpus', default=2, type=int)
    parser.add_argument('--num_sanity_val_steps', default=1, type=int)
    parser.add_argument('--gradient_clip_val', default=5, type=float)
    # ! #

    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6,
                        type=float, help='weight decay')
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--fp16', default=False, type=str2bool, help='FP16')
    parser.add_argument('--sharded_ddp', default=False,
                        type=str2bool, help='fairscale sharded ddp')

    # model args
    parser.add_argument('--dim', default=256, type=int,
                        help='dimension. dimension must be divisible by number of heads.')
    parser.add_argument('--depth', default=4, type=int, help='layers')
    parser.add_argument('--heads', default=4, type=int, help='heads')
    # projection matrix에 의해 head별 차원수: dim_head -> nb_fetures
    parser.add_argument('--dim_head', default=64, type=int,
                        help='dim of head. inner_dim = dim_head * heads')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float)

    parser.add_argument('--local_attn_heads', default=0, type=int,
                        help='if n heads are local attention, heads-n others are global performers.')
    parser.add_argument('--local_window_size', default=256,
                        type=int, help='window size of local attention')
    parser.add_argument('--causal', default=True,
                        type=str2bool, help='auto-regressive or not')
    parser.add_argument('--causal_clm', default='conditioned_causal', choices=[
                        'conditioned_causal', 'causal'], type=str, help='auto-regressive or not')

    parser.add_argument('--nb_features', default=64, type=int,
                        help='number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head.')
    parser.add_argument('--feature_redraw_interval', default=1000, type=int,
                        help='how frequently to redraw the projection matrix, the more frequent, the slower the training')
    parser.add_argument('--reversible', default=False, type=str2bool,
                        help='reversible layers, from Reformer paper. Works only when sharded_ddp=True')
    parser.add_argument('--ff_chunks', default=1, type=int,
                        help='chunk feedforward layer, from Reformer paper')
    parser.add_argument('--ff_glu', default=False, type=str2bool,
                        help='use GLU variant for feedforward')  # !!!!!#
    parser.add_argument('--emb_dropout', default=0.1,
                        type=float, help='embedding dropout')
    parser.add_argument('--ff_dropout', default=0.1,
                        type=float, help='feedforward dropout')
    parser.add_argument('--attn_dropout', default=0.1,
                        type=float, help='post-attn dropout')
    parser.add_argument('--use_scalenorm', default=False, type=str2bool,
                        help='use scale norm, from Transformers without Tears paper')
    parser.add_argument('--use_rezero', default=False, type=str2bool,
                        # scalenorm, rezero, layernorm 중 한가지만 사용 가능.
                        help='use rezero, from Rezero is all you need paper')
    parser.add_argument('--tie_embed', default=False, type=str2bool,
                        help='multiply final embeddings with token weights for logits')
    parser.add_argument('--rotary_position_emb', default=False, type=str2bool,
                        help='use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding')

    parser.add_argument('--attn_type', default='conditioned_noncuda',
                        choices=['noncuda', 'cuda', 'conditioned_noncuda', 'conditioned_cuda'], type=str)
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--FAVOR', action='store_true')
    parser.add_argument('--generalized_attention', action='store_true',
                        help='defaults to softmax approximation, but can be set to True for generalized attention')

    args = parser.parse_args()

    start = datetime.datetime.now()
    TODAY = str(datetime.date.today().strftime('%Y%m%d'))
    NOW = str(start.strftime('_%Hh%Mm'))
    print("\n")
    pl.seed_everything(args.seed, workers=True)

    if args.vqgan == 256:
        args.vqgan_model_path = '/home/edlab/dylee/mimic/mimic-vqgan/mimiccxr_vqgan1024_reso256/checkpoints/last.ckpt'
        args.vqgan_config_path = '/home/edlab/dylee/mimic/mimic-vqgan/mimiccxr_vqgan1024_reso256/configs/2021-07-05T10-23-24-project.yaml'
        args.codebook_indices_path = '/home/edlab/dylee/mimic/mimic-vqgan/codebook_indices/mimiccxr_vqgan1024_codebook_indices.pickle'
    elif args.vqgan == 512:
        args.vqgan_model_path = '/home/edlab/dylee/mimic/mimic-vqgan/mimiccxr_vqgan1024_reso512/checkpoints/last.ckpt'
        args.vqgan_config_path = '/home/edlab/dylee/mimic/mimic-vqgan/mimiccxr_vqgan1024_reso512/configs/2021-12-17T08-58-54-project.yaml'
        args.codebook_indices_path = '/home/edlab/dylee/mimic/mimic-vqgan/codebook_indices/mimiccxr_vqgan1024_res512_codebook_indices.pickle'
    else:
        raise ValueError(
            "Our vqgan have resloution only one between 256 and 512")

    tokenizer = ByteLevelBPETokenizer(
        args.vocab_file,
        args.merge_file,
    )
    tokenizer.add_special_tokens(
        ["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ("[SOS]", tokenizer.token_to_id("[SOS]")),
    )
    # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
    tokenizer.enable_truncation(max_length=args.max_text_len)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(
        "[PAD]"), pad_token="[PAD]", max_length=args.max_text_len)

    dsclass = partial(
        CXRDataset,
        img_root_dir=args.img_root_dir,
        text_root_dir=args.text_root_dir,
        vqgan_model_path=args.vqgan_model_path,
        vqgan_config_path=args.vqgan_config_path,
        codebook_indices_path=args.codebook_indices_path,
        vqgan=args.vqgan,
        max_img_num=args.max_img_num,
        max_text_len=args.max_text_len,
        tokenizer=tokenizer,
        target_count=args.target_count,
        target_view=args.target_view,
        under_sample=args.under_sample,
    )

    train_ds = dsclass(args.train_meta_file)
    val_ds = dsclass(args.val_meta_file)
    test_ds = dsclass(args.test_meta_file)

    dm = CXRDataModule(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # add
    args.num_tokens = train_ds.text_vocab_size  # NOTE: text vocab size 14526
    args.img_vocab_size = train_ds.img_vocab_size
    args.num_img_tokens = train_ds.img_vocab_size + \
        (2 * 4) + train_ds.max_img_num  # NOTE: img vocab size + num img pad  1024
    args.max_seq_len = train_ds.img_len * \
        train_ds.max_img_num + train_ds.max_text_len
    args.max_text_len = train_ds.max_text_len
    args.max_img_len = train_ds.img_len * train_ds.max_img_num
    args.max_img_num = train_ds.max_img_num
    args.img_len = train_ds.img_len
    args.condition_len = args.max_text_len + \
        (train_ds.img_len * (train_ds.max_img_num - 1))
    args.img_fmap_size = int(train_ds.img_fmap_size)
    print("\n condition_len ", args.condition_len, "\n")
    print(" max_seq_len ", args.max_seq_len, "\n")
    print(" num_img_tokens ", args.num_img_tokens, "\n")
    print(" max_img_len ", args.max_img_len, "\n")
    print(" num_tokens(text_vocab_size) ", args.num_tokens, "\n")
    print(" max_text_len ", args.max_text_len, "\n")

    kargs_i2t = {
        'num_tokens': args.num_tokens,
        'img_vocab_size': args.img_vocab_size,
        'num_img_tokens': args.num_img_tokens,
        'max_seq_len': args.max_seq_len,
        'max_img_len': args.max_img_len,
        'max_img_num': args.max_img_num,
        'img_len': args.img_len,
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'dim_head': args.dim_head,
        'local_attn_heads': args.local_attn_heads,
        'local_window_size': args.local_window_size,
        'causal': args.causal,
        # if greater than 0 and causal=True, conditoned causal LM works.
        'condition_len': args.condition_len,
        'attn_type': args.attn_type,
        'nb_features': args.nb_features,
        'feature_redraw_interval': args.feature_redraw_interval,
        'reversible': args.reversible,
        'ff_chunks': args.ff_chunks,
        'ff_glu': args.ff_glu,
        'emb_dropout': args.emb_dropout,
        'ff_dropout': args.ff_dropout,
        'attn_dropout': args.attn_dropout,
        'generalized_attention': args.generalized_attention,
        'kernel_fn': nn.ReLU(),  # used only when args.generalized_attention = True
        'use_scalenorm': args.use_scalenorm,
        'use_rezero': args.use_rezero,
        'tie_embed': args.tie_embed,
        'rotary_position_emb': args.rotary_position_emb,
        'img_fmap_size': args.img_fmap_size,
        'FAVOR': args.FAVOR,
    }

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    base_path = "/home/edlab/dylee/scaleup_transformer/t2i_Performers/"
    folder_path = "t2i_" + str(args.causal_clm) + "_" + str(args.max_img_num) + "of" + str(args.target_count) + "_" + TODAY + NOW +\
        "_d" + str(args.dim) + "_l" + str(args.depth) + "_h" + str(args.heads) + "_" + str(args.attn_type) + "_res" + str(args.vqgan) +\
        "_lr_" + str(args.lr)
    if args.transformer:
        if args.FAVOR:
            if args.generalized_attention:
                save_dirs = base_path + folder_path + "_trans_gen"
            else:
                save_dirs = base_path + folder_path + "_trans_fav"
        else:
            save_dirs = base_path + folder_path + "_transformer"
    else:
        if args.generalized_attention:
            save_dirs = base_path + folder_path + "_perf_gen"
        else:
            save_dirs = base_path + folder_path + "_perf_fav"

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    if LOCAL_RANK == 0 and not args.test:
        os.makedirs(save_dirs, exist_ok=True)

    if not args.transformer:
        model = PerformerLightning_t2i(
            lr=args.lr,
            weight_decay=args.weight_decay,
            tokenizer=tokenizer,
            pad_token_idx=tokenizer.token_to_id("[PAD]"),
            sos_token_idx=tokenizer.token_to_id("[SOS]"),
            eos_token_idx=tokenizer.token_to_id("[EOS]"),
            beam_size=args.beam_size,
            save_dir=save_dirs,
            **kargs_i2t,
        )
    else:
        model = TransformerLightning_t2i(
            lr=args.lr,
            weight_decay=args.weight_decay,
            tokenizer=tokenizer,
            pad_token_idx=tokenizer.token_to_id("[PAD]"),
            sos_token_idx=tokenizer.token_to_id("[SOS]"),
            eos_token_idx=tokenizer.token_to_id("[EOS]"),
            beam_size=args.beam_size,
            save_dir=save_dirs,
            causal_trans=args.causal_clm,
            **kargs_i2t,
        )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dirs,
        verbose=True,
        save_last=True,
        save_top_k=args.save_top_k,
        monitor='val_loss',
        mode='min',
    )

    lr_callback = LearningRateMonitor(
        logging_interval="step",
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='max'
    )
    # profiler = AdvancedProfiler()

    trainer_args = {
        'callbacks': [checkpoint_callback, lr_callback],
        'max_epochs': args.n_epochs,
        'gpus': args.n_gpus,
        'accelerator': 'ddp',
        'num_sanity_val_steps': args.num_sanity_val_steps,
        # catches any bugs in your validation without having to wait for the first validation check.
    }

    if args.reload_ckpt_dir:
        model = model.load_from_checkpoint(args.reload_ckpt_dir)
        # trainer_args['resume_from_checkpoint'] = args.reload_ckpt_dir

    # instrument experiment with W&B
    args.wandb_name = NOW
    wandb_logger = WandbLogger(entity='sut', project='t2i_Performer_' + TODAY, name=args.wandb_name, log_model=True, config=args)

    if (args.fp16 is True and args.sharded_ddp is True):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, precision=16, plugins='ddp_sharded',
                             gradient_clip_val=args.gradient_clip_val)

    elif (args.fp16 is True and args.sharded_ddp is False):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, precision=16, plugins=DDPPlugin(find_unused_parameters=True),
                             # , accumulate_grad_batches=args.accumulate_grad_batches)
                             gradient_clip_val=args.gradient_clip_val)

    elif (args.fp16 is False and args.sharded_ddp is True):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins='ddp_sharded',
                             gradient_clip_val=args.gradient_clip_val)

    elif (args.fp16 is False and args.sharded_ddp is False):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins=DDPPlugin(find_unused_parameters=True),
                             # , accumulate_grad_batches=args.accumulate_grad_batches)
                             gradient_clip_val=args.gradient_clip_val, profiler="simple")
        # trainer = pl.Trainer(**trainer_args,
        #                      logger=wandb_logger,
        #                      checkpoint_callback=checkpoint_callback,
        #                      gradient_clip_val=0.5,
        #                      accumulate_grad_batches=16
        #                     #  profile=True
        #                      )
        # # trainer = pl.Trainer(**trainer_args, logger=wandb_logger, gradient_clip_val=0.5, accumulate_grad_batches=16)

    # log gradients and model topology
    wandb_logger.watch(model)

    if not args.test:
        trainer.fit(model, datamodule=dm)

        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins=DDPPlugin(find_unused_parameters=True),
                             gradient_clip_val=args.gradient_clip_val, profiler="simple", limit_train_batches=0, limit_val_batches=0)
        trainer.test(model, test_dataloaders=dm.test_dataloader())

    else:
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins=DDPPlugin(find_unused_parameters=True),
                             gradient_clip_val=args.gradient_clip_val, profiler="simple", limit_train_batches=0, limit_val_batches=0)
        trainer.test(model, test_dataloaders=dm.test_dataloader())
