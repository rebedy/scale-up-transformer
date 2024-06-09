import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
# from pytorch_lightning.utilities.distributed import rank_zero_only

import csv
import time
import os
import math

from nltk.translate.bleu_score import corpus_bleu
# from cal_metric import get_label_metric_v4
from transformer_pytorch.transformer_pytorch import TransformerLM_i2t, TransformerLM_Protein, TransformerLM_OneBillionWords
# from transformer_pytorch.mlm_pytorch import MLM
from performer_pytorch.performer_pytorch import PerformerLM_i2t, PerformerLM_Protein, PerformerLM_OneBillionWords


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
class PerformerLightning_i2t(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None, pad_token_idx=0, sos_token_idx=1, eos_token_idx=2, save_dir="", **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_i2t = PerformerLM_i2t(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer

    def forward(self, images, texts):
        logit = self.performerLM_i2t(images, texts)
        return logit

    # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        images, texts = batch['images'], batch['texts']
        starttime = time.monotonic()
        # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        logit = self(images, texts)
        endtime = time.monotonic()

        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        _time = math.log2(endtime - starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        # torch.cuda.empty_cache()

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('time', _time, on_step=True, on_epoch=True, sync_dist=True)
        self.log('peak_mem', _peak_mem, on_step=True,
                 on_epoch=True, sync_dist=True)

        output = {
            'batch_idx': batch_idx,
            'loss': loss,
            'time': _time,
            'peak_mem': _peak_mem
        }
        return output

    def training_step_end(self, training_step_outputs, *args, **kwargs):
        if self.trainer.is_global_zero:
            if training_step_outputs["batch_idx"] == 100:
                print("\ntime ", round(training_step_outputs["time"], 3))
                print("peak_mem ", round(training_step_outputs["peak_mem"], 3))
            else:
                pass
        return training_step_outputs

        # def training_epoch_end(self, training_step_outputs):
        #     gathered_outputs = self.all_gather(training_step_outputs)
        #     if self.trainer.is_global_zero:
        #         forward_time = torch.mean(gathered_outputs['time'][0])
        #         backward_time = torch.mean(gathered_outputs['time'][1])
        #         peak_mem = torch.mean(max(gathered_outputs['peak_mem']))
        #         total_train_loss = torch.mean(gathered_outputs['loss'])
        #         self.log("forward_time", forward_time)
        #         self.log("backward_time", backward_time)
        #         self.log("peak_mem", peak_mem)
        #         self.log("total_train_loss", total_train_loss)

    def validation_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts = batch['img_paths'], batch[
            'study_id'], batch['images'], batch['texts']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        self.log('val_loss', loss)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        gen_texts = self.performerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
        )
        pad_size = (0, texts.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'val_loss': loss
        }
        return output

    # validation_step_outputs: list  # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
    def validation_epoch_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        if self.trainer.is_global_zero:
            self.log("val_loss_epoch", total_val_loss)

        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['gen_text'])
        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for out in gathered_validation_step_outputs:
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            GT_decoded_texts = []
            gen_decoded_texts = []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = self.tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(
                    gen_text_i, skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            # calculate BLEU
            references = []
            candidates = []
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                reference = [gt_decoded_text_i.split(' ')]
                candidate = gen_decoded_text_i.split(' ')
                references.append(reference)
                candidates.append(candidate)
            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates,
                                weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates,
                                weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates,
                                weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print('\n\n')
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_eval_' + str(round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_eval_' + str(round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')

            f_gt = open(GT_REPORTS_PATH, 'w')
            wr_gt = csv.writer(f_gt)

            f_gen = open(GEN_REPORTS_PATH, 'w')
            wr_gen = csv.writer(f_gen)

            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])

            f_gt.close()
            f_gen.close()
            print("GEN_reports_eval saved.")
            time.sleep(0.5)

            """
            # run labeler
            time.sleep(0.5)
            LABELED_GT_REPORTS_PATH = os.path.join(dirpath, 'labeled_GT_reports_temp.csv')
            LABELED_GEN_REPORTS_PATH = os.path.join(dirpath, 'labeled_GEN_reports_temp.csv')
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/dylee/__workspace/scale-up-transformer/chexpert-labeler \
                && export PYTHONPATH=/home/dylee/__workspace/scale-up-transformer/chexpert-labelerNegBio:$PYTHONPATH \
                && python /home/dylee/__workspace/scale-up-transformer/chexpert-labeler/label.py \
                --reports_path {os.path.abspath(GT_REPORTS_PATH)} \
                --output_path {os.path.abspath(LABELED_GT_REPORTS_PATH)}',
                shell=True,
                executable='/bin/bash',
                )
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {os.path.abspath(GEN_REPORTS_PATH)} \
                --output_path {os.path.abspath(LABELED_GEN_REPORTS_PATH)}',
                shell=True,
                executable='/bin/bash',
                )

            # calculate metrics
            time.sleep(0.5)
            _, _, _, metric_all, _, _, _, _ = get_label_metric_v4(
                hypothesis = LABELED_GEN_REPORTS_PATH,
                reference = LABELED_GT_REPORTS_PATH,
            )
            print(f"(micro) accuracy, precision, recall, f1 for all : {metric_all[0]:.3f}, {metric_all[1]:.3f}, {metric_all[2]:.3f}, {metric_all[3]:.3f}")
            self.log("val_acc", metric_all[0])
            self.log("val_precision", metric_all[1])
            self.log("val_recall", metric_all[2])
            self.log("val_f1", metric_all[3])
            """

    def test_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts = batch['img_paths'], batch[
            'study_id'], batch['images'], batch['texts']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        # self.log('test_loss', loss)

        gen_texts = self.performerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
        )
        pad_size = (0, texts.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'test_loss': loss
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        self.log('test_loss', total_test_loss)

        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['gen_text'])
        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for out in gathered_test_step_outputs:
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        # -> total_gen_text, total_GT_text: [testset_size, max_text_len]

        if self.global_rank == 0:
            GT_decoded_texts = []
            gen_decoded_texts = []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = self.tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(
                    gen_text_i, skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            # calculate BLEU
            references = []
            candidates = []
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                reference = [gt_decoded_text_i.split(' ')]
                candidate = gen_decoded_text_i.split(' ')
                references.append(reference)
                candidates.append(candidate)
            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates,
                                weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates,
                                weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates,
                                weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print('\n\n')
            print(f'(test)Cumulative 1-gram: {bleu1:.3f}')
            print(f'(test)Cumulative 2-gram: {bleu2:.3f}')
            print(f'(test)Cumulative 3-gram: {bleu3:.3f}')
            print(f'(test)Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)

            # save csv files for labeler

            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')

            f_gt = open(GT_REPORTS_PATH, 'w')
            wr_gt = csv.writer(f_gt)

            f_gen = open(GEN_REPORTS_PATH, 'w')
            wr_gen = csv.writer(f_gen)

            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])

            f_gt.close()
            f_gen.close()

            """
            # run labeler
            time.sleep(0.5)
            LABELED_GT_REPORTS_PATH = os.path.join(dirpath, 'labeled_GT_reports_temp.csv')
            LABELED_GEN_REPORTS_PATH = os.path.join(dirpath, 'labeled_GEN_reports_temp.csv')
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {os.path.abspath(GT_REPORTS_PATH)} \
                --output_path {os.path.abspath(LABELED_GT_REPORTS_PATH)}',
                shell=True,
                executable='/bin/bash',
                )
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {os.path.abspath(GEN_REPORTS_PATH)} \
                --output_path {os.path.abspath(LABELED_GEN_REPORTS_PATH)}',
                shell=True,
                executable='/bin/bash',
                )

            # calculate metrics
            time.sleep(0.5)
            _, _, _, metric_all, _, _, _, _ = get_label_metric_v4(
                hypothesis = LABELED_GEN_REPORTS_PATH,
                reference = LABELED_GT_REPORTS_PATH,
            )
            print(f"(test)(micro) accuracy, precision, recall, f1 for all : {metric_all[0]:.3f}, {metric_all[1]:.3f}, {metric_all[2]:.3f}, {metric_all[3]:.3f}")
            self.log("test_acc", metric_all[0])
            self.log("test_precision", metric_all[1])
            self.log("test_recall", metric_all[2])
            self.log("test_f1", metric_all[3])
            """

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.performerLM_i2t.image_pos_emb.named_parameters():
            wd_params.add(p)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        # return {"optimizer": optimizer, "monitor": "val_loss"}


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
class TransformerLightning_i2t(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None,
                 pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_trans='conditioned_causal',
                 **kargs):
        super().__init__()
        self.kargs = kargs
        self.transformerLM_i2t = TransformerLM_i2t(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir
        self.causal = causal_trans

        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer

    def forward(self, images, texts, view):
        logit = self.transformerLM_i2t(images, texts, view, causal=self.causal)
        return logit

    def training_step(self, batch,
                      # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
                      batch_idx):

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        images, texts, view = batch['images'], batch['texts'], batch['view_position']
        fw_starttime = time.monotonic()
        # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        logit = self(images, texts, view)
        fw_endtime = time.monotonic()
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)

        # bw_starttime = time.monotonic()
        # self.manual_backward(loss)
        # # self.manual_backward(loss, opt, retain_graph=True)
        # opt.step()
        # sch.step()
        # bw_endtime = time.monotonic()

        _fw_time = math.log2(fw_endtime - fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('fw_time', _fw_time, on_step=True,
                 on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'batch_idx': batch_idx,
            'loss': loss,
            'fw_time': _fw_time,
            'peak_mem': _peak_mem
        }
        return output

    # def on_before_zero_grad(self, optimizer):
    # # def on_before_backward(self):
    #     bw_starttime = time.monotonic()
    #     self.log('bw_starttime', bw_starttime, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    #     output = {'bw_starttime': bw_starttime,}
    #     return output

    # def on_after_backward(self):
    #     bw_endtime = time.monotonic()
    #     self.log('bw_endtime', bw_endtime, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    #     output = {'bw_endtime': bw_endtime,}
    #     return output

    # def training_batch_end(self, training_step_outputs, *args, **kwargs):
    #     if self.trainer.is_global_zero:

    #         print("\n")
    #         print("logged_metrics : ", self.trainer.logged_metrics)
    #         print("callback_metrics : ", self.trainer.callback_metrics)
    #         exit()
    #         fw_time = training_step_outputs["fw_time"]
    #         cb_metrics = self.trainer.callback_metrics
    #         bw_time = math.log2(cb_metrics['bw_endtime']-cb_metrics['bw_starttime'])

    #     # output = {
    #     #     'fw_time': fw_time,
    #     #     'bw_time': bw_time,
    #     #     }

    #     return training_step_outputs#, output

    # def training_epoch_end(self, training_step_outputs):

    #     gathered_outputs = self.all_gather(training_step_outputs)
    #     # print(gathered_outputs['time'])

    #     if self.trainer.is_global_zero:
    #         # print("\n")
    #         # print("logged_metrics : ", self.trainer.logged_metrics)
    #         # print("callback_metrics : ", self.trainer.callback_metrics)
    #         # print("\n", gathered_outputs)

    #         cb_metrics = self.trainer.callback_metrics
    #         # fw_time = math.log2(cb_metrics['fw_endtime']-cb_metrics['fw_starttime'])
    #         bw_time = math.log2(cb_metrics['bw_endtime']-cb_metrics['bw_starttime'])

    #         print("\n", gathered_outputs)
    #         print(len(gathered_outputs))
    #         print(gathered_outputs[0])
    #         time_tot = []
    #         for i in range(len(gathered_outputs)):
    #             time_tot.append(gathered_outputs[0]['fw_time'][0])
    #         print(time_tot)
        # print(np.mean(time_tot))

        # forward_time = torch.mean(gathered_outputs[0]['time'][0])
        # backward_time = torch.mean(gathered_outputs['time'][1])
        # peak_mem = torch.mean(max(gathered_outputs['peak_mem']))
        # total_train_loss = torch.mean(gathered_outputs['loss'])
        # self.log("forward_time", forward_time)
        # self.log("backward_time", backward_time)
        # self.log("peak_mem", peak_mem)
        # self.log("total_train_loss", total_train_loss)

    def validation_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts, view = batch['img_paths'], batch[
            'study_id'], batch['images'], batch['texts'], batch['view_position']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]

        logit = self(images, texts, view)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        self.log('val_loss', loss)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        gen_texts = self.transformerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            view,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        pad_size = (0, texts.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'img_paths': img_paths,
            'val_loss': loss
        }
        return output

    # validation_step_outputs: list  # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
    def validation_epoch_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        if self.trainer.is_global_zero:
            self.log("val_loss_epoch", total_val_loss)

        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['gen_text'])
        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for out in gathered_validation_step_outputs:
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            GT_decoded_texts = []
            gen_decoded_texts = []
            # img_paths = []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = self.tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(
                    gen_text_i, skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            # calculate BLEU
            references = []
            candidates = []
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                reference = [gt_decoded_text_i.split(' ')]
                candidate = gen_decoded_text_i.split(' ')
                references.append(reference)
                candidates.append(candidate)
            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates,
                                weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates,
                                weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates,
                                weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print('\n\n')
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            # save csv files for labeler

            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_eval_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_eval_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')

            f_gt = open(GT_REPORTS_PATH, 'w')
            wr_gt = csv.writer(f_gt)

            f_gen = open(GEN_REPORTS_PATH, 'w')
            wr_gen = csv.writer(f_gen)

            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])

            f_gt.close()
            f_gen.close()
            print("GEN_reports_eval saved.")
            time.sleep(0.5)

            """
            # run labeler
            time.sleep(0.5)
            LABELED_GT_REPORTS_PATH = os.path.join(dirpath, 'labeled_GT_reports_temp.csv')
            LABELED_GEN_REPORTS_PATH = os.path.join(dirpath, 'labeled_GEN_reports_temp.csv')
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {GT_REPORTS_PATH} \
                --output_path {LABELED_GT_REPORTS_PATH}',
                shell=True,
                executable='/bin/bash',
            )
            print('ran labeled gt reports')

            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {GEN_REPORTS_PATH} \
                --output_path {LABELED_GEN_REPORTS_PATH}',
                shell=True,
                executable='/bin/bash',
            )


            # calculate metrics
            time.sleep(0.5)
            _, _, _, metric_all, _, _, _, _ = get_label_metric_v4(
                hypothesis=LABELED_GEN_REPORTS_PATH,
                reference=LABELED_GT_REPORTS_PATH,
            )
            print(
                f"(micro) accuracy, precision, recall, f1 for all : {metric_all[0]:.3f}, {metric_all[1]:.3f}, {metric_all[2]:.3f}, {metric_all[3]:.3f}")
            self.log("val_acc", metric_all[0])
            self.log("val_precision", metric_all[1])
            self.log("val_recall", metric_all[2])
            self.log("val_f1", metric_all[3])
            """

    def test_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts, view = batch['img_paths'], batch[
            'study_id'], batch['images'], batch['texts'], batch['view_position']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        logit = self(images, texts, view)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        # self.log('test_loss', loss)

        gen_texts = self.transformerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        pad_size = (0, texts.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'test_loss': loss,
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        if self.trainer.is_global_zero:
            self.log('test_loss_epoch', total_test_loss)

        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['gen_text'])
        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for out in gathered_test_step_outputs:
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        # -> total_gen_text, total_GT_text: [testset_size, max_text_len]

        if self.global_rank == 0:
            GT_decoded_texts = []
            gen_decoded_texts = []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = self.tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(
                    gen_text_i, skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            # calculate BLEU
            references = []
            candidates = []
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                reference = [gt_decoded_text_i.split(' ')]
                candidate = gen_decoded_text_i.split(' ')
                references.append(reference)
                candidates.append(candidate)
            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates,
                                weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates,
                                weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates,
                                weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print('\n\n')
            print(f'(test)Cumulative 1-gram: {bleu1:.3f}')
            print(f'(test)Cumulative 2-gram: {bleu2:.3f}')
            print(f'(test)Cumulative 3-gram: {bleu3:.3f}')
            print(f'(test)Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)

            # save csv files for labeler

            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_twst_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')

            f_gt = open(GT_REPORTS_PATH, 'w')
            wr_gt = csv.writer(f_gt)

            f_gen = open(GEN_REPORTS_PATH, 'w')
            wr_gen = csv.writer(f_gen)

            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])

            f_gt.close()
            f_gen.close()
            print('Finished Saving Files')

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.transformerLM_i2t.image_pos_emb.named_parameters():
            wd_params.add(p)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
class CLMplusMLMLightning_i2t(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None,
                 mask_token_idx=4, pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_clm='conditioned_causal', causal_mlm='full',
                 **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_i2t = TransformerLM_i2t(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir

        self.causal_clm = causal_clm
        self.causal_mlm = causal_mlm

        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer

    def forward(self, images, texts, causal):
        logit = self.performerLM_i2t(images, texts, causal)
        return logit

    def training_step(self, batch,
                      # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
                      batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        images, texts, masked_texts, masked_label = batch['images'], batch[
            'texts'], batch['masked_texts'], batch['masked_label']
        condition_len = self.kargs['condition_len']

        starttime_1 = time.monotonic()
        # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        logit_clm = self(images, texts, causal=self.causal_clm)
        endtime_1 = time.monotonic()
        # logit_clm = logit_clm[:, condition_len:-1].transpose(1,2)#.reshape(-1, logit_clm.size(-1))
        # target = texts[:, 1:]#.reshape(-1)
        # .reshape(-1, logit_clm.size(-1))
        logit_clm = logit_clm[:, condition_len:-1, :].transpose(1, 2)
        target = texts[:, 1:]
        loss_clm = cross_entropy(
            logit_clm, target, ignore_index=self.pad_token_idx)
        _time_1 = math.log2(endtime_1 - starttime_1)
        _peak_mem_1 = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        starttime_2 = time.monotonic()
        # -> [B, img_len * max_img_num + max_text_len, num_tokens]
        logit_mlm = self(images, masked_texts, causal=self.causal_mlm)
        endtime_2 = time.monotonic()
        # .reshape(-1, logit_mlm.size(-1)) #[max_text_len, num_tokens=14526]
        logit_mlm = logit_mlm[:, condition_len:-1, :].transpose(1, 2)
        target = masked_label[:, 1:]  # .reshape(-1)
        loss_mlm = cross_entropy(logit_mlm, target, ignore_index=-100)

        _time_2 = math.log2(endtime_2 - starttime_2)
        _peak_mem_2 = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        # torch.cuda.empty_cache()

        loss = loss_clm + loss_mlm

        self.log('train_loss_clm', loss_clm, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('train_loss_mlm', loss_mlm, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('time_clm', _time_1, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('peak_mem_clm', _peak_mem_1, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('time_mlm', _time_2, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('peak_mem_mlm', _peak_mem_2, on_step=True,
                 on_epoch=True, sync_dist=True)

        output = {
            'batch_idx': batch_idx,
            'loss': loss,
            'time_clm': _time_1,
            'peak_mem_clm': _peak_mem_1,
            'time_mlm': _time_2,
            'peak_mem_mlm': _peak_mem_2,
        }

        return output

    def training_step_end(self, training_step_outputs, *args, **kwargs):
        if self.trainer.is_global_zero:
            if training_step_outputs["batch_idx"] == 300:
                print("\ntime_clm ", round(
                    training_step_outputs["time_clm"], 3))
                print("peak_mem_clm ", round(
                    training_step_outputs["peak_mem_clm"], 3))
                print("time_mlm ", round(training_step_outputs["time_mlm"], 3))
                print("peak_mem_mlm ", round(
                    training_step_outputs["peak_mem_mlm"], 3))
            else:
                pass
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        img_paths, study_ids = batch['img_paths'], batch['study_id']
        images, texts, masked_texts, masked_label = batch['images'], batch[
            'texts'], batch['masked_texts'], batch['masked_label']
        condition_len = self.kargs['condition_len']

        # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        logit_clm = self(images, texts, causal=self.causal_clm)
        # .reshape(-1, logit_clm.size(-1))
        logit_clm = logit_clm[:, condition_len:-1, :].transpose(1, 2)
        target = texts[:, 1:]
        loss_clm = cross_entropy(
            logit_clm, target, ignore_index=self.pad_token_idx)

        # -> [B, img_len * max_img_num + max_text_len, num_tokens]
        logit_mlm = self(images, masked_texts, causal=self.causal_mlm)
        # .reshape(-1, logit_mlm.size(-1)) #[max_text_len, num_tokens=14526]
        logit_mlm = logit_mlm[:, condition_len:-1, :].transpose(1, 2)
        target = masked_label[:, 1:]  # .reshape(-1)
        loss_mlm = cross_entropy(logit_mlm, target, ignore_index=-100)

        loss = loss_clm + loss_mlm
        self.log('val_loss_clm', loss_clm)
        self.log('val_loss_mlm', loss_mlm)
        self.log('val_loss', loss)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        gen_texts = self.performerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal_clm,
        )
        pad_size = (0, texts.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'img_paths': img_paths,
            'val_loss': loss
        }
        return output

    # validation_step_outputs: list  # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
    def validation_epoch_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        if self.trainer.is_global_zero:
            self.log("val_loss_epoch", total_val_loss)

        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['gen_text'])
        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for out in gathered_validation_step_outputs:
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            GT_decoded_texts = []
            gen_decoded_texts = []
            # img_paths = []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = self.tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(
                    gen_text_i, skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            # calculate BLEU
            references = []
            candidates = []
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                reference = [gt_decoded_text_i.split(' ')]
                candidate = gen_decoded_text_i.split(' ')
                references.append(reference)
                candidates.append(candidate)
            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates,
                                weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates,
                                weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates,
                                weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print('\n\n')
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            # save csv files for labeler

            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_eval_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_eval_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')

            f_gt = open(GT_REPORTS_PATH, 'w')
            wr_gt = csv.writer(f_gt)

            f_gen = open(GEN_REPORTS_PATH, 'w')
            wr_gen = csv.writer(f_gen)

            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])

            f_gt.close()
            f_gen.close()
            print("GEN_reports_eval saved.")
            time.sleep(0.5)

    def test_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts = batch['img_paths'], batch[
            'study_id'], batch['images'], batch['texts']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        condition_len = self.kargs['condition_len']

        logit = self(images, texts)
        logit = logit[:, condition_len:-1, :].transpose(1, 2)
        target = texts[:, 1:]
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        self.log('test_loss', loss)

        gen_texts = self.performerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
        )
        pad_size = (0, texts.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'test_loss': loss,
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        self.log('test_loss', total_test_loss)

        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['gen_text'])
        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for out in gathered_test_step_outputs:
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        # -> total_gen_text, total_GT_text: [testset_size, max_text_len]

        if self.global_rank == 0:
            GT_decoded_texts = []
            gen_decoded_texts = []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = self.tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(
                    gen_text_i, skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            # calculate BLEU
            references = []
            candidates = []
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                reference = [gt_decoded_text_i.split(' ')]
                candidate = gen_decoded_text_i.split(' ')
                references.append(reference)
                candidates.append(candidate)
            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates,
                                weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates,
                                weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates,
                                weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print('\n\n')
            print(f'(test)Cumulative 1-gram: {bleu1:.3f}')
            print(f'(test)Cumulative 2-gram: {bleu2:.3f}')
            print(f'(test)Cumulative 3-gram: {bleu3:.3f}')
            print(f'(test)Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)

            # save csv files for labeler

            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_twst_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')

            f_gt = open(GT_REPORTS_PATH, 'w')
            wr_gt = csv.writer(f_gt)

            f_gen = open(GEN_REPORTS_PATH, 'w')
            wr_gen = csv.writer(f_gen)

            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])

            f_gt.close()
            f_gen.close()
            print('Finished Saving Files')

            """
            # run labeler
            time.sleep(0.5)
            LABELED_GT_REPORTS_PATH = os.path.join(dirpath, 'labeled_GT_reports_test.csv')
            LABELED_GEN_REPORTS_PATH = os.path.join(dirpath, 'labeled_GEN_reports_test.csv')
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {GT_REPORTS_PATH} \
                --output_path {LABELED_GT_REPORTS_PATH}',
                shell=True,
                executable='/bin/bash',
            )
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
                && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
                && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
                --reports_path {GEN_REPORTS_PATH} \
                --output_path {LABELED_GEN_REPORTS_PATH}',
                shell=True,
                executable='/bin/bash',
            )

            # calculate metrics
            time.sleep(0.5)
            _, _, _, metric_all, _, _, _, _ = get_label_metric_v4(
                hypothesis=LABELED_GEN_REPORTS_PATH,
                reference=LABELED_GT_REPORTS_PATH,
            )
            print(
                f"(test)(micro) accuracy, precision, recall, f1 for all : {metric_all[0]:.3f}, {metric_all[1]:.3f}, {metric_all[2]:.3f}, {metric_all[3]:.3f}")
            self.log("test_acc", metric_all[0])
            self.log("test_precision", metric_all[1])
            self.log("test_recall", metric_all[2])
            self.log("test_f1", metric_all[3])
            """

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.performerLM_i2t.image_pos_emb.named_parameters():
            wd_params.add(p)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class PerformerLightning_protein(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None, pad_token_idx=0, sos_token_idx=1, eos_token_idx=2, **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_protein = PerformerLM_Protein(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.tokenizer = tokenizer
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_hyperparameters()

    def forward(self, data):
        logit = self.performerLM_protein(data)
        return logit

    def training_step(self, batch, batch_idx):
        data = batch
        # starttime = time.monotonic()
        logit = self(data)
        # endtime = time.monotonic()
        # print(f'forward time: {math.log2(endtime-starttime)}')
        logit = logit.reshape(-1, 30)
        data = data.reshape(-1)
        loss = cross_entropy(logit, data, ignore_index=0)
        self.log('train_loss', loss)
        # self.log('train_forward', endtime-starttime)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        gen_proteins = self.performerLM_protein.generate_proteins(data)

        # loss = cross_entropy(logit.reshape(-1, 30), data.reshape(-1), ignore_index=0)
        output = {
            'label': data,
            'predict': gen_proteins,
            # 'val_loss': loss
        }

        data = data.reshape(-1)
        gen_proteins = gen_proteins.reshape(-1)

        acc = (data == gen_proteins).sum()
        acc = acc / gen_proteins.size(0)
        self.log('val_acc', acc)

        return output

    def validation_step_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        # total_val_loss = torch.mean(gathered_validation_step_outputs['val_loss'])
        # self.log('val_loss', total_val_loss)

        # total_predict = gathered_validation_step_outputs['predict'].reshape(-1, 30)
        # total_predict = total_predict.argmax(-1)

        total_label = gathered_validation_step_outputs['label'].reshape(-1)
        total_predict = gathered_validation_step_outputs['predict'].reshape(-1)

        acc = (total_predict == total_label).sum()
        acc = acc / total_predict.size(0)
        self.log('val_total_acc', acc)

    def test_step(self, batch, batch_idx):
        data = batch
        gen_proteins = self.performerLM_protein.generate_proteins(data)

        # loss = cross_entropy(logit.reshape(-1, 30), data.reshape(-1), ignore_index=0)
        output = {
            'label': data,
            'predict': gen_proteins,
            # 'val_loss': loss
        }
        return output

    def test_step_end(self, test_step_outputs):
        gathered_validation_step_outputs = self.all_gather(test_step_outputs)

        # total_val_loss = torch.mean(gathered_validation_step_outputs['val_loss'])
        # self.log('val_loss', total_val_loss)

        # total_predict = gathered_validation_step_outputs['predict'].reshape(-1, 30)
        # total_predict = total_predict.argmax(-1)

        total_label = gathered_validation_step_outputs['label'].reshape(-1)
        total_predict = gathered_validation_step_outputs['predict'].reshape(-1)

        acc = (total_predict == total_label).sum()
        acc = acc / total_predict.size(0)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


class PerformerLightning_OneBillionWords(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_OneBillionWords = PerformerLM_OneBillionWords(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, data):
        logit = self.performerLM_OneBillionWords(data)
        return logit

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        # starttime = time.monotonic()
        logit = self(data)
        # endtime = time.monotonic()
        # print(f'forward time: {math.log2(endtime-starttime)}')
        logit = logit.reshape(-1, 30522)
        label = label.reshape(-1)
        loss = cross_entropy(logit, label, ignore_index=-100, reduction='none')

        ppl = torch.exp(loss[torch.where(loss != 0.)]).mean()

        mask_count = (data == 103).sum()
        loss = (loss.sum() / mask_count)

        metric = {
            'train_loss': loss,
            'train_ppl': ppl
        }
        self.log_dict(metric)
        # self.log('train_forward', endtime-starttime)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']

        logit = self(data)
        logit = logit.reshape(-1, 30522)
        label = label.reshape(-1)
        loss = cross_entropy(logit, label, ignore_index=-100, reduction='none')

        return loss

    def validation_step_end(self, validation_step_outputs):
        eval_loss = self.all_gather(validation_step_outputs)

        ppl = torch.exp(eval_loss[torch.where(eval_loss != 0.)]).mean()

        mask_count = (eval_loss != 0).sum()
        loss = (eval_loss.sum() / mask_count)

        metric = {
            'eval_loss': loss,
            'eval_ppl': ppl
        }

        self.log_dict(metric)

    def test_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']

        logit = self(data)
        logit = logit.reshape(-1, 30522)
        label = label.reshape(-1)
        loss = cross_entropy(logit, label, ignore_index=-100, reduction='none')
        return loss

    def test_step_end(self, test_step_outputs):
        test_loss = self.all_gather(test_step_outputs)

        ppl = torch.exp(test_loss[torch.where(test_loss != 0.)]).mean()

        mask_count = (test_loss != 0.).sum()
        loss = (test_loss.sum() / mask_count)

        metric = {
            'test_loss': loss,
            'test_ppl': ppl
        }

        self.log_dict(metric)

    def configure_optimizers(self):
        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "eval_loss"}


class TransformerLightning_protein(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None, pad_token_idx=0, sos_token_idx=1, eos_token_idx=2, **kargs):
        super().__init__()
        self.kargs = kargs
        self.transformerLM_protein = TransformerLM_Protein(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.tokenizer = tokenizer
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_hyperparameters()

    def forward(self, data):
        logit = self.transformerLM_protein(data)
        return logit

    def training_step(self, batch, batch_idx):
        data = batch
        # starttime = time.monotonic()
        logit = self(data)
        # endtime = time.monotonic()
        # print(f'forward time: {math.log2(endtime-starttime)}')
        logit = logit.reshape(-1, 30)
        data = data.reshape(-1)
        loss = cross_entropy(logit, data)
        self.log('train_loss', loss)
        # self.log('train_forward', endtime-starttime)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        gen_proteins = self.transformerLM_protein.generate_proteins(data)

        output = {
            'label': data,
            'predict': gen_proteins,
        }

        return output

    def validation_step_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_label = gathered_validation_step_outputs['label'].reshape(-1)
        total_predict = gathered_validation_step_outputs['predict'].reshape(-1)

        acc = (total_predict == total_label).sum()
        acc = acc / total_predict.size(0)
        self.log('val_total_acc', acc)

    def test_step(self, batch, batch_idx):
        data = batch
        gen_proteins = self.transformerLM_protein.generate_proteins(data)

        output = {
            'label': data,
            'predict': gen_proteins,
        }

        return output

    def test_step_end(self, test_step_outputs):
        gathered_validation_step_outputs = self.all_gather(test_step_outputs)

        total_label = gathered_validation_step_outputs['label'].reshape(-1)
        total_predict = gathered_validation_step_outputs['predict'].reshape(-1)

        acc = (total_predict == total_label).sum()
        acc = acc / total_predict.size(0)
        self.log('test_total_acc', acc)

    def configure_optimizers(self):
        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


###################################################################
class TransformerLightning_OneBillionWords(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, **kargs):
        super().__init__()
        self.kargs = kargs
        self.transformerLM_OneBillionWords = TransformerLM_OneBillionWords(
            **kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, data):
        logit = self.transformerLM_OneBillionWords(data)
        return logit

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        logit = self(data)
        logit = logit.reshape(-1, 30522)
        label = label.reshape(-1)
        loss = cross_entropy(logit, label, ignore_index=-100, reduction='none')

        ppl = torch.exp(loss[torch.where(loss != 0.)]).mean()

        mask_count = (data == 103).sum()
        loss = (loss.sum() / mask_count)

        metric = {
            'train_loss': loss,
            'train_ppl': ppl
        }
        self.log_dict(metric)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']

        logit = self(data)
        logit = logit.reshape(-1, 30522)
        label = label.reshape(-1)
        loss = cross_entropy(logit, label, ignore_index=-100, reduction='none')
        return loss

    def validation_step_end(self, validation_step_outputs):
        eval_loss = self.all_gather(validation_step_outputs)

        ppl = torch.exp(eval_loss[torch.where(eval_loss != 0.)]).mean()

        mask_count = (eval_loss != 0).sum()
        loss = (eval_loss.sum() / mask_count)

        metric = {
            'eval_loss': loss,
            'eval_ppl': ppl
        }

        self.log_dict(metric)

    def test_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']

        logit = self(data)
        logit = logit.reshape(-1, 30522)
        label = label.reshape(-1)
        loss = cross_entropy(logit, label, ignore_index=-100, reduction='none')
        return loss

    def test_step_end(self, test_step_outputs):
        test_loss = self.all_gather(test_step_outputs)

        ppl = torch.exp(test_loss[torch.where(test_loss != 0.)]).mean()

        mask_count = (test_loss != 0.).sum()
        loss = (test_loss.sum() / mask_count)

        metric = {
            'test_loss': loss,
            'test_ppl': ppl
        }

        self.log_dict(metric)

    def configure_optimizers(self):
        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        no_wd_params = all_params - wd_params
        wd_params = list(wd_params)
        no_wd_params = list(no_wd_params)

        optimizer_grouped_parameters = [
            {
                "params": wd_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_wd_params,
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "eval_loss"}
