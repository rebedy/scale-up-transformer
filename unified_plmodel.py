# from datetime import timedelta
import numpy as np
# from optax import adam
import pytorch_lightning as pl
# from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.nn.functional import cross_entropy
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch
import torch.nn as nn
import csv
import time
# from cal_metric import get_label_metric_v4
import os
import math
import random

from nltk.translate.bleu_score import corpus_bleu
from transformer_pytorch.transformer_unified import TransformerLM_unified, TransformerLM_unified222

random.seed(42)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
class TransformerLightning_unified(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01,
                 pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_trans='conditioned_causal',
                 beam_size=1, **kargs):
        super().__init__()
        self.kargs = kargs
        self.transformerLM_unified = TransformerLM_unified(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir
        self.causal = causal_trans
        self.beam_size = beam_size
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['tokenizer'])

    def forward(self, img1, img2, txt, modes, view):
        logit = self.transformerLM_unified(
            img1, img2, txt, modes, view, causal=self.causal)
        return logit

    def training_step(self, batch, batch_idx):
        # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img1, img2, txt, modes, view = batch['img1'], batch[
            'img2'], batch['txt'], batch['modes'], batch['view_position']
        fw_starttime = time.monotonic()
        logit = self(img1, img2, txt, modes, view)
        fw_endtime = time.monotonic()

        loss = None

        if modes[-1][0] == 'txt':
            images = torch.cat((img1, img2), dim=1)  # -> [B, seq_len]
            b, condition_len = images.shape
            logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
            target = txt[:, 1:].reshape(-1)
            loss = cross_entropy(
                logit, target, ignore_index=self.pad_token_idx)

        elif modes[-1][0] == 'img1':
            b, n_img1 = img1.shape
            logit = logit[:, (-n_img1 - 1):-1].reshape(-1, logit.size(-1))
            target = img1.reshape(-1)
            loss = cross_entropy(logit, target)

        elif modes[-1][0] == 'img2':
            b, n_img2 = img2.shape
            logit = logit[:, (-n_img2 - 1):-1].reshape(-1, logit.size(-1))
            target = img2.reshape(-1)
            loss = cross_entropy(logit, target)

        else:
            raise ValueError

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

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img_paths, study_ids = batch['img_paths'], batch['study_id']
        img1, img2, txt, view = batch['img1'], batch['img2'], batch['txt'], batch['view_position']

        loss, gen_texts, gen_images, images = None, None, None, None

        # ! # TXT
        modes_txt = [['img1'], ['img2']]
        np.random.shuffle(modes_txt)  # , len(modes))
        modes_txt.append(['txt'])
        logit = self(img1, img2, txt, modes_txt, view)
        images = torch.cat((img1, img2), dim=1)  # -> [B, seq_len]
        b, condition_len = images.shape
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        target = txt[:, 1:].reshape(-1)
        txt_loss = cross_entropy(
            logit, target, ignore_index=self.pad_token_idx)
        gen_texts = self.transformerLM_unified.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            img1,
            img2,
            view,
            modes_txt,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        pad_size = (0, txt.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        # ! # IMG1
        modes_img1 = [['txt'], ['img2']]
        np.random.shuffle(modes_img1)
        modes_img1.append(['img1'])
        logit = self(img1, img2, txt, modes_img1, view)
        images1 = img1
        b, n_img1 = img1.shape
        logit = logit[:, (-n_img1 - 1):-1].reshape(-1, logit.size(-1))
        target = img1.reshape(-1)
        img1_loss = cross_entropy(logit, target)
        gen_images1 = self.transformerLM_unified.generate_image(
            txt,
            img2,
            view,
            modes_img1,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        # ! # IMG2
        modes_img2 = [['txt'], ['img1']]
        np.random.shuffle(modes_img2)
        modes_img2.append(['img2'])
        logit = self(img1, img2, txt, modes_img2, view)
        images2 = img2
        b, n_img2 = img2.shape
        logit = logit[:, (-n_img2 - 1):-1].reshape(-1, logit.size(-1))
        target = img2.reshape(-1)
        img2_loss = cross_entropy(logit, target)
        gen_images2 = self.transformerLM_unified.generate_image(
            txt,
            img1,
            view,
            modes_img2,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        self.log('val_txt_loss', txt_loss)
        self.log('val_img1_loss', img1_loss)
        self.log('val_img2_loss', img2_loss)

        loss = txt_loss + img1_loss + img2_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        self.log('val_peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': images1,
            'GT_image2': images2,
            'gen_image1': gen_images1,
            'gen_image2': gen_images2,
            'img_paths': img_paths,
            'val_loss': loss,
            'val_txt_loss': txt_loss,
            'val_img1_loss': img1_loss,
            'val_img2_loss': img2_loss,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'modes_img2': modes_img2,
        }

        return output

    # validation_step_outputs: list
    def validation_epoch_end(self, validation_step_outputs):
        # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing
        tokenizer = ByteLevelBPETokenizer(
            'BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer.add_special_tokens(
            ["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
        )
        # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(
            "[PAD]"), pad_token="[PAD]", length=256)
        # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다
        ##########

        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        total_val_txt_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_txt_loss'])
        total_val_img1_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_img1_loss'])
        total_val_img2_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_img2_loss'])
        if self.trainer.is_global_zero:
            self.log("total_val_loss", total_val_loss)
            self.log("val_txt_loss_epoch", total_val_txt_loss)
            self.log("val_img1_loss_epoch", total_val_img1_loss)
            self.log("val_img2_loss_epoch", total_val_img2_loss)

        img_paths = gathered_validation_step_outputs[0]['img_paths']
        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])

        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for i, out in enumerate(gathered_validation_step_outputs):
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
            # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            # ! # For generated images
            torch.save(gathered_validation_step_outputs, os.path.join(
                self.save_dir, str(self.current_epoch) + '_eval_output.pt'))

            # ! # For generated texts
            GT_decoded_texts, gen_decoded_texts = [], []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = tokenizer.decode(
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
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(self.save_dir, str(self.current_epoch) + '_GT_reports_eval_' + str(
                round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, str(self.current_epoch) + '_GEN_reports_eval_' + str(
                round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            IMG_PATHS = os.path.join(self.save_dir, str(self.current_epoch) + '_IMG_paths_eval_' + str(round(
                bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            f_gt = open(GT_REPORTS_PATH, 'a')
            wr_gt = csv.writer(f_gt)
            f_gen = open(GEN_REPORTS_PATH, 'a')
            wr_gen = csv.writer(f_gen)
            f_img = open(IMG_PATHS, 'a')
            wr_img = csv.writer(f_img)
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])
            for img_paths_i in img_paths:
                wr_img.writerow([img_paths_i])
            f_gt.close()
            f_gen.close()
            f_img.close()
            print("GEN_reports_eval saved.")
            print('\n\n')

        time.sleep(0.5)

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img_paths, study_ids = batch['img_paths'], batch['study_id']
        img1, img2, txt, view = batch['img1'], batch['img2'], batch['txt'], batch['view_position']

        # ! # TXT
        modes_txt = [['img1'], ['img2']]
        np.random.shuffle(modes_txt)  # , len(modes))
        modes_txt.append(['txt'])
        logit = self(img1, img2, txt, modes_txt, view)
        images = torch.cat((img1, img2), dim=1)  # -> [B, seq_len]
        b, condition_len = images.shape
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        target = txt[:, 1:].reshape(-1)
        txt_loss = cross_entropy(
            logit, target, ignore_index=self.pad_token_idx)
        gen_texts = self.transformerLM_unified.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            img1,
            img2,
            view,
            modes_txt,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        pad_size = (0, txt.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        # ! # IMG1
        modes_img1 = [['txt'], ['img2']]
        np.random.shuffle(modes_img1)
        modes_img1.append(['img1'])
        logit = self(img1, img2, txt, modes_img1, view)
        images1 = img1
        b, n_img1 = img1.shape
        logit = logit[:, (-n_img1 - 1):-1].reshape(-1, logit.size(-1))
        target = img1.reshape(-1)
        img1_loss = cross_entropy(logit, target)
        gen_images1 = self.transformerLM_unified.generate_image(
            txt,
            img2,
            view,
            modes_img1,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        # ! # IMG2
        modes_img2 = [['txt'], ['img1']]
        np.random.shuffle(modes_img2)
        modes_img2.append(['img2'])
        logit = self(img1, img2, txt, modes_img2, view)
        images2 = img2
        b, n_img2 = img2.shape
        logit = logit[:, (-n_img2 - 1):-1].reshape(-1, logit.size(-1))
        target = img2.reshape(-1)
        img2_loss = cross_entropy(logit, target)
        gen_images2 = self.transformerLM_unified.generate_image(
            txt,
            img1,
            view,
            modes_img2,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        self.log('test_txt_loss', txt_loss)
        self.log('test_img1_loss', img1_loss)
        self.log('test_img2_loss', img2_loss)

        loss = txt_loss + img1_loss + img2_loss
        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)

        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        self.log('test_peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': images1,
            'GT_image2': images2,
            'gen_image1': gen_images1,
            'gen_image2': gen_images2,
            'img_paths': img_paths,
            'test_loss': loss,
            'test_txt_loss': txt_loss,
            'test_img1_loss': img1_loss,
            'test_img2_loss': img2_loss,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'modes_img2': modes_img2,
        }

        return output

    def test_epoch_end(self, test_step_outputs):
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing
        tokenizer = ByteLevelBPETokenizer(
            'BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer.add_special_tokens(
            ["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
        )
        # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(
            "[PAD]"), pad_token="[PAD]", max_length=256)
        # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다
        ##########

        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        total_test_txt_loss = torch.mean(
            gathered_test_step_outputs[0]['test_txt_loss'])
        total_test_img1_loss = torch.mean(
            gathered_test_step_outputs[0]['test_img1_loss'])
        total_test_img2_loss = torch.mean(
            gathered_test_step_outputs[0]['test_img2_loss'])
        if self.trainer.is_global_zero:
            self.log("total_test_loss", total_test_loss)
            self.log("test_txt_loss_epoch", total_test_txt_loss)
            self.log("test_img1_loss_epoch", total_test_img1_loss)
            self.log("test_img2_loss_epoch", total_test_img2_loss)

        img_paths = gathered_test_step_outputs[0]['img_paths']
        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])

        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for i, out in enumerate(gathered_test_step_outputs):
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
            # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            # ! # For generated images
            torch.save(gathered_test_step_outputs, os.path.join(
                self.save_dir, 'test_output.pt'))

            # ! # For generated texts
            GT_decoded_texts, gen_decoded_texts = [], []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = tokenizer.decode(
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
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)
            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            IMG_PATHS = os.path.join(self.save_dir, 'IMG_paths_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            f_gt = open(GT_REPORTS_PATH, 'a')
            wr_gt = csv.writer(f_gt)
            f_gen = open(GEN_REPORTS_PATH, 'a')
            wr_gen = csv.writer(f_gen)
            f_img = open(IMG_PATHS, 'a')
            wr_img = csv.writer(f_img)
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])
            for img_paths_i in img_paths:
                wr_img.writerow([img_paths_i])
            f_gt.close()
            f_gen.close()
            f_img.close()
            print("GEN_reports_test saved.")
            print('\n\n')

        time.sleep(0.5)

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.transformerLM_unified.image_pos_emb.named_parameters():
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

        # optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)

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


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
class TransformerLightning_unified2(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01,
                 pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_trans='conditioned_causal',
                 beam_size=1, **kargs):
        super().__init__()
        self.kargs = kargs
        self.transformerLM_unified = TransformerLM_unified222(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir
        self.causal = causal_trans
        self.beam_size = beam_size
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['tokenizer'])

    def forward(self, img1, img2, txt, modes, view):
        logit = self.transformerLM_unified(
            img1, img2, txt, modes, view, causal=self.causal)
        return logit

    def training_step(self, batch, batch_idx):
        # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img1, img2, txt, modes, view = batch['img1'], batch[
            'img2'], batch['txt'], batch['modes'], batch['view_position']
        fw_starttime = time.monotonic()
        logit = self(img1, img2, txt, modes, view)
        fw_endtime = time.monotonic()

        loss = None

        if modes[-1][0] == 'txt':
            images = torch.cat((img1, img2), dim=1)  # -> [B, seq_len]
            b, condition_len = images.shape
            logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
            target = txt[:, 1:].reshape(-1)
            loss = cross_entropy(
                logit, target, ignore_index=self.pad_token_idx)

        elif modes[-1][0] == 'img1':
            b, n_img1 = img1.shape
            logit = logit[:, (-n_img1 - 1):-1].reshape(-1, logit.size(-1))
            target = img1.reshape(-1)
            loss = cross_entropy(logit, target)

        elif modes[-1][0] == 'img2':
            b, n_img2 = img2.shape
            logit = logit[:, (-n_img2 - 1):-1].reshape(-1, logit.size(-1))
            target = img2.reshape(-1)
            loss = cross_entropy(logit, target)

        else:
            raise ValueError

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

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img_paths, study_ids = batch['img_paths'], batch['study_id']
        img1, img2, txt, view = batch['img1'], batch['img2'], batch['txt'], batch['view_position']

        loss, gen_texts, gen_images, images = None, None, None, None

        # ! # TXT
        modes_txt = [['img1'], ['img2']]
        np.random.shuffle(modes_txt)  # , len(modes))
        modes_txt.append(['txt'])
        logit = self(img1, img2, txt, modes_txt, view)
        images = torch.cat((img1, img2), dim=1)  # -> [B, seq_len]
        b, condition_len = images.shape
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        target = txt[:, 1:].reshape(-1)
        txt_loss = cross_entropy(
            logit, target, ignore_index=self.pad_token_idx)
        gen_texts = self.transformerLM_unified.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            img1,
            img2,
            view,
            modes_txt,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        pad_size = (0, txt.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        # ! # IMG1
        modes_img1 = [['txt'], ['img2']]
        np.random.shuffle(modes_img1)
        modes_img1.append(['img1'])
        logit = self(img1, img2, txt, modes_img1, view)
        images1 = img1
        b, n_img1 = img1.shape
        logit = logit[:, (-n_img1 - 1):-1].reshape(-1, logit.size(-1))
        target = img1.reshape(-1)
        img1_loss = cross_entropy(logit, target)
        gen_images1 = self.transformerLM_unified.generate_image(
            txt,
            img2,
            view,
            modes_img1,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        # ! # IMG2
        modes_img2 = [['txt'], ['img1']]
        np.random.shuffle(modes_img2)
        modes_img2.append(['img2'])
        logit = self(img1, img2, txt, modes_img2, view)
        images2 = img2
        b, n_img2 = img2.shape
        logit = logit[:, (-n_img2 - 1):-1].reshape(-1, logit.size(-1))
        target = img2.reshape(-1)
        img2_loss = cross_entropy(logit, target)
        gen_images2 = self.transformerLM_unified.generate_image(
            txt,
            img1,
            view,
            modes_img2,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        self.log('val_txt_loss', txt_loss)
        self.log('val_img1_loss', img1_loss)
        self.log('val_img2_loss', img2_loss)

        loss = txt_loss + img1_loss + img2_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        self.log('val_peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': images1,
            'GT_image2': images2,
            'gen_image1': gen_images1,
            'gen_image2': gen_images2,
            'img_paths': img_paths,
            'val_loss': loss,
            'val_txt_loss': txt_loss,
            'val_img1_loss': img1_loss,
            'val_img2_loss': img2_loss,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'modes_img2': modes_img2,
        }

        return output

    # validation_step_outputs: list
    def validation_epoch_end(self, validation_step_outputs):
        # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing
        tokenizer = ByteLevelBPETokenizer(
            'BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer.add_special_tokens(
            ["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
        )
        # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(
            "[PAD]"), pad_token="[PAD]", max_length=256)
        # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다
        ##########

        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        total_val_txt_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_txt_loss'])
        total_val_img1_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_img1_loss'])
        total_val_img2_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_img2_loss'])
        if self.trainer.is_global_zero:
            self.log("total_val_loss", total_val_loss)
            self.log("val_txt_loss_epoch", total_val_txt_loss)
            self.log("val_img1_loss_epoch", total_val_img1_loss)
            self.log("val_img2_loss_epoch", total_val_img2_loss)

        img_paths = gathered_validation_step_outputs[0]['img_paths']
        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_validation_step_outputs[0]['GT_text'])

        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for i, out in enumerate(gathered_validation_step_outputs):
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
            # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            # ! # For generated images
            torch.save(gathered_validation_step_outputs, os.path.join(
                self.save_dir, str(self.current_epoch) + '_eval_output.pt'))

            # ! # For generated texts
            GT_decoded_texts, gen_decoded_texts = [], []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = tokenizer.decode(
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
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(self.save_dir, str(self.current_epoch) + '_GT_reports_eval_' + str(
                round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, str(self.current_epoch) + '_GEN_reports_eval_' + str(
                round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            IMG_PATHS = os.path.join(self.save_dir, str(self.current_epoch) + '_IMG_paths_eval_' + str(round(
                bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            f_gt = open(GT_REPORTS_PATH, 'a')
            wr_gt = csv.writer(f_gt)
            f_gen = open(GEN_REPORTS_PATH, 'a')
            wr_gen = csv.writer(f_gen)
            f_img = open(IMG_PATHS, 'a')
            wr_img = csv.writer(f_img)
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])
            for img_paths_i in img_paths:
                wr_img.writerow([img_paths_i])
            f_gt.close()
            f_gen.close()
            f_img.close()
            print("GEN_reports_eval saved.")
            print('\n\n')

        time.sleep(0.5)

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img_paths, study_ids = batch['img_paths'], batch['study_id']
        img1, img2, txt, view = batch['img1'], batch['img2'], batch['txt'], batch['view_position']

        # ! # TXT
        modes_txt = [['img1'], ['img2']]
        np.random.shuffle(modes_txt)  # , len(modes))
        modes_txt.append(['txt'])
        logit = self(img1, img2, txt, modes_txt, view)
        images = torch.cat((img1, img2), dim=1)  # -> [B, seq_len]
        b, condition_len = images.shape
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        target = txt[:, 1:].reshape(-1)
        txt_loss = cross_entropy(
            logit, target, ignore_index=self.pad_token_idx)
        gen_texts = self.transformerLM_unified.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            img1,
            img2,
            view,
            modes_txt,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        pad_size = (0, txt.size(-1) - gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant',
                          self.pad_token_idx)  # -> tensor[B, max_text_len]

        # ! # IMG1
        modes_img1 = [['txt'], ['img2']]
        np.random.shuffle(modes_img1)
        modes_img1.append(['img1'])
        logit = self(img1, img2, txt, modes_img1, view)
        images1 = img1
        b, n_img1 = img1.shape
        logit = logit[:, (-n_img1 - 1):-1].reshape(-1, logit.size(-1))
        target = img1.reshape(-1)
        img1_loss = cross_entropy(logit, target)
        gen_images1 = self.transformerLM_unified.generate_image(
            txt,
            img2,
            view,
            modes_img1,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        # ! # IMG2
        modes_img2 = [['txt'], ['img1']]
        np.random.shuffle(modes_img2)
        modes_img2.append(['img2'])
        logit = self(img1, img2, txt, modes_img2, view)
        images2 = img2
        b, n_img2 = img2.shape
        logit = logit[:, (-n_img2 - 1):-1].reshape(-1, logit.size(-1))
        target = img2.reshape(-1)
        img2_loss = cross_entropy(logit, target)
        gen_images2 = self.transformerLM_unified.generate_image(
            txt,
            img1,
            view,
            modes_img2,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        self.log('test_txt_loss', txt_loss)
        self.log('test_img1_loss', img1_loss)
        self.log('test_img2_loss', img2_loss)

        loss = txt_loss + img1_loss + img2_loss
        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)

        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        self.log('test_peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': images1,
            'GT_image2': images2,
            'gen_image1': gen_images1,
            'gen_image2': gen_images2,
            'img_paths': img_paths,
            'test_loss': loss,
            'test_txt_loss': txt_loss,
            'test_img1_loss': img1_loss,
            'test_img2_loss': img2_loss,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'modes_img2': modes_img2,
        }

        return output

    def test_epoch_end(self, test_step_outputs):
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing
        tokenizer = ByteLevelBPETokenizer(
            'BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer.add_special_tokens(
            ["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
        )
        # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(
            "[PAD]"), pad_token="[PAD]", max_length=256)
        # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다
        ##########

        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        total_test_txt_loss = torch.mean(
            gathered_test_step_outputs[0]['test_txt_loss'])
        total_test_img1_loss = torch.mean(
            gathered_test_step_outputs[0]['test_img1_loss'])
        total_test_img2_loss = torch.mean(
            gathered_test_step_outputs[0]['test_img2_loss'])
        if self.trainer.is_global_zero:
            self.log("total_test_loss", total_test_loss)
            self.log("test_txt_loss_epoch", total_test_txt_loss)
            self.log("test_img1_loss_epoch", total_test_img1_loss)
            self.log("test_img2_loss_epoch", total_test_img2_loss)

        img_paths = gathered_test_step_outputs[0]['img_paths']
        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(
            gathered_test_step_outputs[0]['GT_text'])

        # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
        for i, out in enumerate(gathered_test_step_outputs):
            GT_text = out['GT_text'].reshape(-1, max_text_len)
            gen_text = out['gen_text'].reshape(-1, max_text_len)
            total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
            total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
            # -> total_gen_text, total_GT_text: [valset_size, max_text_len]

        if self.global_rank == 0:
            # ! # For generated images
            torch.save(gathered_test_step_outputs, os.path.join(
                self.save_dir, 'test_output.pt'))

            # ! # For generated texts
            GT_decoded_texts, gen_decoded_texts = [], []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_text_i = gt_text_i.tolist()
                gen_text_i = gen_text_i.tolist()
                gt_decoded_text_i = tokenizer.decode(
                    gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = tokenizer.decode(
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
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)
            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            IMG_PATHS = os.path.join(self.save_dir, 'IMG_paths_test_' + str(round(bleu1, 3)) + '_' + str(
                round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '.csv')
            f_gt = open(GT_REPORTS_PATH, 'a')
            wr_gt = csv.writer(f_gt)
            f_gen = open(GEN_REPORTS_PATH, 'a')
            wr_gen = csv.writer(f_gen)
            f_img = open(IMG_PATHS, 'a')
            wr_img = csv.writer(f_img)
            for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                wr_gt.writerow([gt_decoded_text_i])
                wr_gen.writerow([gen_decoded_text_i])
            for img_paths_i in img_paths:
                wr_img.writerow([img_paths_i])
            f_gt.close()
            f_gen.close()
            f_img.close()
            print("GEN_reports_test saved.")
            print('\n\n')

        time.sleep(0.5)

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.transformerLM_unified.image_pos_emb.named_parameters():
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

        # optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)

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
