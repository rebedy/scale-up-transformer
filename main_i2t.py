import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from functools import partial
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing # This post-processor takes care of adding the special tokens: a [EOS] token and a [SOS] token
from loader import CXRDataset
from datamodule import CXRDataModule
from i2t_plmodel import PerformerLightning_i2t
from pytorch_lightning.plugins import DDPPlugin
from utils import str2bool

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--train_meta_file', default='metadata/mdvl_mimiccxr_train.csv', type=str)
    parser.add_argument('--val_meta_file', default='metadata/mdvl_mimiccxr_valid.csv', type=str)
    parser.add_argument('--test_meta_file', default='metadata/mdvl_mimiccxr_test.csv', type=str)
    parser.add_argument('--img_root_dir', default='/home/edlab/dylee/physionet.org/files/mimic-cxr-jpg/2.0.0/files', type=str)
    parser.add_argument('--text_root_dir', default='/home/edlab/dylee/physionet.org/files/mimic-cxr-jpg/2.0.0/preprocessed_reports_mdvl', type=str)
    parser.add_argument('--vqgan_model_path', default='/home/edlab/dylee/vqgan_cxr/mimiccxr_vqgan1024/checkpoints/last.ckpt', type=str)
    parser.add_argument('--vqgan_config_path', default='/home/edlab/dylee/vqgan_cxr/mimiccxr_vqgan1024/configs/2021-07-05T10-23-24-project.yaml', type=str)
    parser.add_argument('--codebook_indices_path', default='/home/edlab/dylee/codebook_indices/mimiccxr_vqgan1024_codebook_indices.pickle', type=str)
    parser.add_argument('--max_img_num', default=1, type=int, help='must be less than or equal to target_count')
    parser.add_argument('--max_text_len', default=256, type=int)
    parser.add_argument('--vocab_file', default='BBPE_tokenizer/vocab.json', type=str)
    parser.add_argument('--merge_file', default='BBPE_tokenizer/merges.txt', type=str)
    parser.add_argument('--target_count', default=1, type=int)
    parser.add_argument('--target_view', default=['AP', 'AP AXIAL', 'PA', 'LATERAL', 'LL', ''], nargs='+', type=str)
    parser.add_argument('--use_first_img', default=False, type=str2bool)

    # training args
    parser.add_argument('--reload_ckpt_dir', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--n_gpus', default=2, type=int)
    parser.add_argument('--save_top_k', default=5, type=int)
    parser.add_argument('--fp16', default=True, type=str2bool, help='FP16')
    parser.add_argument('--sharded_ddp', default=False, type=str2bool, help='fairscale sharded ddp')

    # model args
    parser.add_argument('--dim', default=768, type=int, help='dimension. dimension must be divisible by number of heads.')
    parser.add_argument('--depth', default=12, type=int, help='layers')
    parser.add_argument('--heads', default=12, type=int, help='heads')
    parser.add_argument('--dim_head', default=64, type=int, help='dim of head. inner_dim = dim_head * heads')  # projection matrix에 의해 head별 차원수: dim_head -> nb_fetures
    parser.add_argument('--local_attn_heads', default=0, type=int, help='if n heads are local attention, heads-n others are global performers.')
    parser.add_argument('--local_window_size', default=256, type=int, help='window size of local attention')
    parser.add_argument('--causal', default=True, type=str2bool, help='auto-regressive or not')
    parser.add_argument('--nb_features', default=64, type=int, help='number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head.')
    parser.add_argument('--feature_redraw_interval', default=1000, type=int, help='how frequently to redraw the projection matrix, the more frequent, the slower the training')
    parser.add_argument('--reversible', default=False, type=str2bool, help='reversible layers, from Reformer paper. Works only when sharded_ddp=True')
    parser.add_argument('--ff_chunks', default=10, type=int, help='chunk feedforward layer, from Reformer paper')
    parser.add_argument('--ff_glu', default=True, type=str2bool, help='use GLU variant for feedforward')
    parser.add_argument('--emb_dropout', default=0.1, type=float, help='embedding dropout')
    parser.add_argument('--ff_dropout', default=0.1, type=float, help='feedforward dropout')
    parser.add_argument('--attn_dropout', default=0.1, type=float, help='post-attn dropout')
    parser.add_argument('--generalized_attention', default=False, type=str2bool, help='defaults to softmax approximation, but can be set to True for generalized attention')
    parser.add_argument('--use_scalenorm', default=False, type=str2bool, help='use scale norm, from Transformers without Tears paper')
    parser.add_argument('--use_rezero', default=False, type=str2bool, help='use rezero, from Rezero is all you need paper')  # scalenorm, rezero, layernorm 중 한가지만 사용 가능.
    parser.add_argument('--tie_embed', default=False, type=str2bool, help='multiply final embeddings with token weights for logits')
    parser.add_argument('--rotary_position_emb', default=False, type=str2bool, help='use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding')

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    tokenizer = ByteLevelBPETokenizer(
        args.vocab_file,
        args.merge_file,
    )
    tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ("[SOS]", tokenizer.token_to_id("[SOS]")),
    )
    tokenizer.enable_truncation(max_length=args.max_text_len)  # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=args.max_text_len)  # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다

    dsclass = partial(
        CXRDataset,
        img_root_dir=args.img_root_dir,
        text_root_dir=args.text_root_dir,
        vqgan_model_path=args.vqgan_model_path,
        vqgan_config_path=args.vqgan_config_path,
        codebook_indices_path=args.codebook_indices_path,
        max_img_num=args.max_img_num,
        max_text_len=args.max_text_len,
        tokenizer=tokenizer,
        target_count=args.target_count,
        target_view=args.target_view,
        use_first_img=args.use_first_img,
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
    args.num_tokens = train_ds.text_vocab_size  # NOTE: text vocab size
    args.num_img_tokens = train_ds.img_vocab_size + train_ds.max_img_num   # NOTE: img vocab size + num img pad
    args.max_seq_len = train_ds.img_len * train_ds.max_img_num + train_ds.max_text_len
    args.max_img_num = train_ds.max_img_num
    args.condition_len = train_ds.img_len * train_ds.max_img_num
    args.img_fmap_size = int(train_ds.img_fmap_size)

    kargs_performerLM_i2t = {
        'num_tokens': args.num_tokens,
        'num_img_tokens': args.num_img_tokens,
        'max_seq_len': args.max_seq_len,
        'max_img_num': args.max_img_num,
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'dim_head': args.dim_head,
        'local_attn_heads': args.local_attn_heads,
        'local_window_size': args.local_window_size,
        'causal': args.causal,
        'condition_len': args.condition_len,        # if greater than 0 and causal=True, conditoned causal LM works.
        'nb_features': args.nb_features,
        'feature_redraw_interval': args.feature_redraw_interval,
        'reversible': args.reversible,
        'ff_chunks': args.ff_chunks,
        'ff_glu': args.ff_glu,
        'emb_dropout': args.emb_dropout,
        'ff_dropout': args.ff_dropout,
        'attn_dropout': args.attn_dropout,
        'generalized_attention': args.generalized_attention,
        'kernel_fn': nn.ReLU(),                    # used only when args.generalized_attention = True
        'use_scalenorm': args.use_scalenorm,
        'use_rezero': args.use_rezero,
        'tie_embed': args.tie_embed,
        'rotary_position_emb': args.rotary_position_emb,
        'img_fmap_size': args.img_fmap_size,
    }

    model = PerformerLightning_i2t(
        lr=args.lr,
        weight_decay=args.weight_decay,
        tokenizer=tokenizer,
        pad_token_idx=tokenizer.token_to_id("[PAD]"),
        sos_token_idx=tokenizer.token_to_id("[SOS]"),
        eos_token_idx=tokenizer.token_to_id("[EOS]"),
        **kargs_performerLM_i2t,
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
        filename="{epoch:06}",
        save_top_k=args.save_top_k,
        monitor='val_loss',
        mode='min',
    )

    lr_callback = LearningRateMonitor(
        logging_interval="step",
    )

    trainer_args = {
        'callbacks': [checkpoint_callback, lr_callback],
        'max_epochs': args.n_epochs,
        'gpus': args.n_gpus,
        'accelerator': 'ddp',
    }

    if args.reload_ckpt_dir:
        trainer_args['resume_from_checkpoint'] = args.reload_ckpt_dir

    # instrument experiment with W&B
    wandb_logger = WandbLogger(project='ScaleUp Transformers', save_dir="./logs", log_model=False, config=args)

    if (args.fp16 is True and args.sharded_ddp is True):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, precision=16, plugins='ddp_sharded', gradient_clip_val=0.5)
    elif (args.fp16 is True and args.sharded_ddp is False):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, precision=16, plugins=DDPPlugin(find_unused_parameters=False), gradient_clip_val=0.5)
    elif (args.fp16 is False and args.sharded_ddp is True):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins='ddp_sharded', gradient_clip_val=0.5)
    elif (args.fp16 is False and args.sharded_ddp is False):
        trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins=DDPPlugin(find_unused_parameters=False), gradient_clip_val=0.5)

    # log gradients and model topology
    wandb_logger.watch(model)

    trainer.fit(model, datamodule=dm)

    trainer.test()
