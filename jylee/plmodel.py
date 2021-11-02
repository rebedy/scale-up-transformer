import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from performer_pytorch import PerformerLM_i2t, PerformerLM_Protein
from transformer_pytorch.transformer_pytorch import TransformerLM_i2t, TransformerLM_Protein
import torch.nn.functional as F
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
import csv
import subprocess
import time
from cal_metric import get_label_metric_v4
import os



class PerformerLightning_i2t(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None, pad_token_idx=0, sos_token_idx=1, eos_token_idx=2, **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_i2t = PerformerLM_i2t(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.tokenizer = tokenizer
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters()

    def forward(self, images, texts):
        logit = self.performerLM_i2t(images, texts)
        return logit

    def training_step(self, batch, batch_idx): # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
        images, texts = batch['images'], batch['texts'] 
        logit = self(images, texts)    # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts = batch['img_paths'], batch['study_id'], batch['images'], batch['texts']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        # self.log('val_loss', loss)

        gen_texts = self.performerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx = self.sos_token_idx,
            eos_token_idx = self.eos_token_idx,
            pad_token_idx = self.pad_token_idx,
            filter_logits_fn = 'top_p',
            filter_thres = 0.9,
            temperature = 0.7,
            )
        pad_size = (0, texts.size(-1)-gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant', self.pad_token_idx) # -> tensor[B, max_text_len]
        
        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'val_loss': loss
        }
        return output

    def validation_epoch_end(self, validation_step_outputs):  # validation_step_outputs: list  # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
        gathered_validation_step_outputs = self.all_gather(validation_step_outputs)

        total_val_loss = torch.mean(gathered_validation_step_outputs[0]['val_loss'])
        self.log('val_loss', total_val_loss)

        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(gathered_validation_step_outputs[0]['gen_text'])
        for out in gathered_validation_step_outputs: # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]} 
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
                gt_decoded_text_i = self.tokenizer.decode(gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(gen_text_i, skip_special_tokens=True)
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
            bleu2 = corpus_bleu(references, candidates, weights=(1/2, 1/2, 0, 0))
            bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0))
            bleu4 = corpus_bleu(references, candidates, weights=(1/4, 1/4, 1/4, 1/4))
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            # save csv files for labeler
            GT_REPORTS_PATH = 'GT_reports_temp.csv'
            GEN_REPORTS_PATH = 'GEN_reports_temp.csv'
            
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
            LABELED_GT_REPORTS_PATH = 'labeled_GT_reports_temp.csv'
            LABELED_GEN_REPORTS_PATH = 'labeled_GEN_reports_temp.csv'
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd ~/scaleup_transformer/chexpert-labeler \
                && export PYTHONPATH=~/scaleup_transformer/chexpert-labeler/NegBio:$PYTHONPATH \
                && python ~/scaleup_transformer/chexpert-labeler/label.py \
                --reports_path {os.path.abspath(GT_REPORTS_PATH)} \
                --output_path {os.path.abspath(LABELED_GT_REPORTS_PATH)}', 
                shell=True,
                executable='/bin/bash',
                )
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd ~/scaleup_transformer/chexpert-labeler \
                && export PYTHONPATH=~/scaleup_transformer/chexpert-labeler/NegBio:$PYTHONPATH \
                && python ~/scaleup_transformer/chexpert-labeler/label.py \
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
        img_paths, study_ids, images, texts = batch['img_paths'], batch['study_id'], batch['images'], batch['texts']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        # self.log('test_loss', loss)

        gen_texts = self.performerLM_i2t.generate_texts(  # gen_texts: tensor[B, <max_text_len]
            images,
            sos_token_idx = self.sos_token_idx,
            eos_token_idx = self.eos_token_idx,
            pad_token_idx = self.pad_token_idx,
            filter_logits_fn = 'top_p',
            filter_thres = 0.9,
            temperature = 0.7,
            )
        pad_size = (0, texts.size(-1)-gen_texts.size(-1))
        gen_texts = F.pad(gen_texts, pad_size, 'constant', self.pad_token_idx) # -> tensor[B, max_text_len]
        
        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'test_loss': loss
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        total_test_loss = torch.mean(gathered_test_step_outputs[0]['test_loss'])
        self.log('test_loss', total_test_loss)

        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['gen_text'])
        for out in gathered_test_step_outputs: # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]} 
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
                gt_decoded_text_i = self.tokenizer.decode(gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(gen_text_i, skip_special_tokens=True)
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
            bleu2 = corpus_bleu(references, candidates, weights=(1/2, 1/2, 0, 0))
            bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0))
            bleu4 = corpus_bleu(references, candidates, weights=(1/4, 1/4, 1/4, 1/4))
            print(f'(test)Cumulative 1-gram: {bleu1:.3f}')
            print(f'(test)Cumulative 2-gram: {bleu2:.3f}')
            print(f'(test)Cumulative 3-gram: {bleu3:.3f}')
            print(f'(test)Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)

            # save csv files for labeler
            GT_REPORTS_PATH = 'GT_reports_temp.csv'
            GEN_REPORTS_PATH = 'GEN_reports_temp.csv'
            
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
            LABELED_GT_REPORTS_PATH = 'labeled_GT_reports_temp.csv'
            LABELED_GEN_REPORTS_PATH = 'labeled_GEN_reports_temp.csv'
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd ~/scaleup_transformer/chexpert-labeler \
                && export PYTHONPATH=~/scaleup_transformer/chexpert-labeler/NegBio:$PYTHONPATH \
                && python ~/scaleup_transformer/chexpert-labeler/label.py \
                --reports_path {os.path.abspath(GT_REPORTS_PATH)} \
                --output_path {os.path.abspath(LABELED_GT_REPORTS_PATH)}', 
                shell=True,
                executable='/bin/bash',
                )
            subprocess.run(
                f'source ~/anaconda3/bin/activate chexpert-label \
                && cd ~/scaleup_transformer/chexpert-labeler \
                && export PYTHONPATH=~/scaleup_transformer/chexpert-labeler/NegBio:$PYTHONPATH \
                && python ~/scaleup_transformer/chexpert-labeler/label.py \
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
        # TODO: 추후 scheduler 변경도 고려해보기


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
        starttime = time.time()
        logit = self(data)
        endtime = time.time()
        print(f'forward time: {endtime-starttime}')
        logit = logit.reshape(-1, 20)
        data = data.reshape(-1)
        loss = cross_entropy(logit, data)
        self.log('train_loss', loss)
        self.log('train_forward', endtime-starttime)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        logit = self(data)
        # print(f'forward time: {endtime - starttime}')
        loss = cross_entropy(logit.reshape(-1, 20), data.reshape(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        data = batch
        logit = self(data)
        loss = cross_entropy(logit.reshape(-1, 20), data.reshape(-1))
        self.log('test_loss', loss)
        return loss

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
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class TransformerLightning_i2t(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None, pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_i2t = TransformerLM_i2t(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.tokenizer = tokenizer
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters()

    def forward(self, images, texts):
        logit = self.performerLM_i2t(images, texts)
        return logit

    def training_step(self, batch,
                      batch_idx):  # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
        images, texts = batch['images'], batch['texts']
        logit = self(images, texts)  # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts = batch['img_paths'], batch['study_id'], batch['images'], batch['texts']
        # img_paths, study_ids: list    images: [B, tot_img_len]   texts: [B, max_text_len]
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx)
        # self.log('val_loss', loss)

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
        gen_texts = F.pad(gen_texts, pad_size, 'constant', self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'val_loss': loss
        }
        return output

    def validation_epoch_end(self, validation_step_outputs):  # validation_step_outputs: list  # DDP에서는 'GPU process별로' validation_step, validation_step_end를 거쳐 validation_step_outputs라는 리스트에 원소로 쌓인다.
        gathered_validation_step_outputs = self.all_gather(validation_step_outputs)

        total_val_loss = torch.mean(gathered_validation_step_outputs[0]['val_loss'])
        self.log('val_loss', total_val_loss)

        max_text_len = gathered_validation_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(gathered_validation_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(gathered_validation_step_outputs[0]['gen_text'])
        for out in gathered_validation_step_outputs:  # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
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
                gt_decoded_text_i = self.tokenizer.decode(gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(gen_text_i, skip_special_tokens=True)
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
            bleu2 = corpus_bleu(references, candidates, weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print(f'Cumulative 1-gram: {bleu1:.3f}')
            print(f'Cumulative 2-gram: {bleu2:.3f}')
            print(f'Cumulative 3-gram: {bleu3:.3f}')
            print(f'Cumulative 4-gram: {bleu4:.3f}')
            self.log("val_BLEU-1", bleu1)
            self.log("val_BLEU-2", bleu2)
            self.log("val_BLEU-3", bleu3)
            self.log("val_BLEU-4", bleu4)

            dirpath = '/home/edlab/jylee/Scaleup/output/TransformerFAVOR_3090'

            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(dirpath, 'GT_reports_temp.csv')
            GEN_REPORTS_PATH = os.path.join(dirpath, 'GEN_reports_temp.csv')

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
            time.sleep(0.5)


            # # run labeler
            # time.sleep(0.5)
            # LABELED_GT_REPORTS_PATH = os.path.join(dirpath, 'labeled_GT_reports_temp.csv')
            # LABELED_GEN_REPORTS_PATH = os.path.join(dirpath, 'labeled_GEN_reports_temp.csv')
            # subprocess.run(
            #     f'source ~/anaconda3/bin/activate chexpert-label \
            #     && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
            #     && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
            #     && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
            #     --reports_path {GT_REPORTS_PATH} \
            #     --output_path {LABELED_GT_REPORTS_PATH}',
            #     shell=True,
            #     executable='/bin/bash',
            # )
            # print('ran labeled gt reports')
            #
            # subprocess.run(
            #     f'source ~/anaconda3/bin/activate chexpert-label \
            #     && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
            #     && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
            #     && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
            #     --reports_path {GEN_REPORTS_PATH} \
            #     --output_path {LABELED_GEN_REPORTS_PATH}',
            #     shell=True,
            #     executable='/bin/bash',
            # )
            #
            #
            # # calculate metrics
            # time.sleep(0.5)
            # _, _, _, metric_all, _, _, _, _ = get_label_metric_v4(
            #     hypothesis=LABELED_GEN_REPORTS_PATH,
            #     reference=LABELED_GT_REPORTS_PATH,
            # )
            # print(
            #     f"(micro) accuracy, precision, recall, f1 for all : {metric_all[0]:.3f}, {metric_all[1]:.3f}, {metric_all[2]:.3f}, {metric_all[3]:.3f}")
            # self.log("val_acc", metric_all[0])
            # self.log("val_precision", metric_all[1])
            # self.log("val_recall", metric_all[2])
            # self.log("val_f1", metric_all[3])
            #

    def test_step(self, batch, batch_idx):
        img_paths, study_ids, images, texts = batch['img_paths'], batch['study_id'], batch['images'], batch['texts']
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
        gen_texts = F.pad(gen_texts, pad_size, 'constant', self.pad_token_idx)  # -> tensor[B, max_text_len]

        output = {
            'GT_text': texts,
            'gen_text': gen_texts,
            'test_loss': loss,
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)


        total_test_loss = torch.mean(gathered_test_step_outputs[0]['test_loss'])
        self.log('test_loss', total_test_loss)

        max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        total_GT_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])
        total_gen_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['gen_text'])
        for out in gathered_test_step_outputs:  # out = {'GT_text': [num_gups, B, max_text_len], 'gen_text': [num_gups, B, max_text_len]}
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
                gt_decoded_text_i = self.tokenizer.decode(gt_text_i, skip_special_tokens=True)
                gen_decoded_text_i = self.tokenizer.decode(gen_text_i, skip_special_tokens=True)
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
            bleu2 = corpus_bleu(references, candidates, weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(references, candidates, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
            print(f'(test)Cumulative 1-gram: {bleu1:.3f}')
            print(f'(test)Cumulative 2-gram: {bleu2:.3f}')
            print(f'(test)Cumulative 3-gram: {bleu3:.3f}')
            print(f'(test)Cumulative 4-gram: {bleu4:.3f}')
            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)

            dirpath = '/home/edlab/jylee/Scaleup/output/TransformerFAVOR_3090'

            # save csv files for labeler
            GT_REPORTS_PATH = os.path.join(dirpath, 'GT_reports_temp.csv')
            GEN_REPORTS_PATH = os.path.join(dirpath, 'GEN_reports_temp.csv')

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


            # # run labeler
            # time.sleep(0.5)
            # LABELED_GT_REPORTS_PATH = os.path.join(dirpath, 'labeled_GT_reports_temp.csv')
            # LABELED_GEN_REPORTS_PATH = os.path.join(dirpath, 'labeled_GEN_reports_temp.csv')
            # subprocess.run(
            #     f'source ~/anaconda3/bin/activate chexpert-label \
            #     && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
            #     && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
            #     && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
            #     --reports_path {GT_REPORTS_PATH} \
            #     --output_path {LABELED_GT_REPORTS_PATH}',
            #     shell=True,
            #     executable='/bin/bash',
            # )
            # subprocess.run(
            #     f'source ~/anaconda3/bin/activate chexpert-label \
            #     && cd /home/jylee/ScaleUp/scaleup/chexpert-labeler \
            #     && export PYTHONPATH=/home/jylee/ScaleUp/scaleup/chexpert-labeler/NegBio:$PYTHONPATH \
            #     && python /home/jylee/ScaleUp/scaleup/chexpert-labeler/label.py \
            #     --reports_path {GEN_REPORTS_PATH} \
            #     --output_path {LABELED_GEN_REPORTS_PATH}',
            #     shell=True,
            #     executable='/bin/bash',
            # )
            #
            # # calculate metrics
            # time.sleep(0.5)
            # _, _, _, metric_all, _, _, _, _ = get_label_metric_v4(
            #     hypothesis=LABELED_GEN_REPORTS_PATH,
            #     reference=LABELED_GT_REPORTS_PATH,
            # )
            # print(
            #     f"(test)(micro) accuracy, precision, recall, f1 for all : {metric_all[0]:.3f}, {metric_all[1]:.3f}, {metric_all[2]:.3f}, {metric_all[3]:.3f}")
            # self.log("test_acc", metric_all[0])
            # self.log("test_precision", metric_all[1])
            # self.log("test_recall", metric_all[2])
            # self.log("test_f1", metric_all[3])
            #

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
        # TODO: 추후 scheduler 변경도 고려해보기


class TransformerLightning_protein(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None, pad_token_idx = 0, sos_token_idx=1, eos_token_idx=2, **kargs):
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
        starttime = time.time()
        logit = self(data)
        endtime = time.time()
        print(f'forward time: {endtime-starttime}')
        logit = logit.reshape(-1, 20)
        data = data.reshape(-1)
        loss = cross_entropy(logit, data)
        self.log('train_loss', loss)
        self.log('train_forward', endtime-starttime)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        logit = self(data)
        # print(f'forward time: {endtime - starttime}')
        loss = cross_entropy(logit.reshape(-1, 20), data.reshape(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        data = batch
        logit = self(data)
        loss = cross_entropy(logit.reshape(-1, 20), data.reshape(-1))
        self.log('test_loss', loss)
        return loss

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
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}