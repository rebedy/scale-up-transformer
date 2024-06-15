from sklearn.metrics import *
# from pytorch_lightning.metrics.classification import Accuracy, AUROC, PrecisionRecallCurve
# from torchmetrics.functional import accuracy, auc

from itertools import cycle
import matplotlib.pyplot as plt
from performer_pytorch.performer_pytorch import PerformerLM_t2i
from transformer_pytorch.transformer_pytorch import TransformerLM_t2i
# from pytorch_lightning.utilities.distributed import rank_zero_only
import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torch.nn as nn
import torch
import os
import math
import time

import numpy as np


np.seterr(divide='ignore', invalid='ignore')


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

#################
# Performer t2i #
#################

class PerformerLightning_t2i(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, save_dir='./', **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_t2i = PerformerLM_t2i(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_dir = save_dir
        self.save_hyperparameters()

    def forward(self, images, texts):
        logit = self.performerLM_t2i(images, texts)
        return logit

    def training_step(self, batch, batch_idx):
        images, texts = batch['images'], batch['texts']
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = images.reshape(-1)
        logit = logit[:, (condition_len - 1):-1].reshape(-1, logit.size(-1))
        # 여기서는 ignore index가 없다. 왜냐면 패딩이 없기 때문에
        loss = cross_entropy(logit, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch['images'], batch['texts']

        logit = self(images, texts)
        condition_len = self.kargs['condition_len']

        target = images.reshape(-1)
        logit = logit[:, (condition_len - 1): -1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target)

        gen_images = self.performerLM_t2i.generate_image(
            texts,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7
        )

        output = {
            'GT_image': images,
            'gen_image': gen_images,
            'val_loss': loss
        }
        return output

    def validation_epoch_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        self.log("val_loss_epoch", total_val_loss)

        if self.global_rank == 0:
            torch.save(gathered_validation_step_outputs,
                       os.path.join(self.save_dir, 'eval_output.pt'))

    def test_step(self, batch, batch_idx):
        images, texts = batch['images'], batch['texts']
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = images.reshape(-1)
        logit = logit[:, (condition_len - 1):-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target)

        gen_images = self.performerLM_t2i.generate_image(
            texts,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7
        )

        output = {
            'GT_image': images,
            'gen_image': gen_images,
            'test_loss': loss
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)
        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        self.log('test_loss', total_test_loss)

        if self.global_rank == 0:
            torch.save(gathered_test_step_outputs, os.path.join(
                self.save_dir, 'test_output.pt'))

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.performerLM_t2i.image_pos_emb.named_parameters():
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
###################
# Transformer t2i #
###################
class TransformerLightning_t2i(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01, tokenizer=None,
                 pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_trans='conditioned_causal',
                 beam_size=1, **kargs):
        super().__init__()
        self.kargs = kargs
        self.transformerLM_t2i = TransformerLM_t2i(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.beam_size = beam_size
        self.save_dir = save_dir
        self.causal = causal_trans

        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer

    def forward(self, images, texts, view):
        logit = self.transformerLM_t2i(images, texts, view, causal=self.causal)
        return logit

    def training_step(self, batch, batch_idx):
        # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        images, texts, view = batch['images'], batch['texts'], batch['view_position']
        condition_len = self.kargs['condition_len']
        img_len = self.kargs['img_len']

        fw_starttime = time.monotonic()
        # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        logit = self(images, texts, view)
        fw_endtime = time.monotonic()

        target = images[:, -img_len:].reshape(-1)
        logit = logit[:, (condition_len - 1):-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target)

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
        img_paths, images, texts, view = batch['img_paths'], batch['images'], batch['texts'], batch['view_position']
        condition_len = self.kargs['condition_len']
        img_len = self.kargs['img_len']

        logit = self(images, texts, view)

        target = images[:, -img_len:].reshape(-1)
        logit = logit[:, (condition_len - 1): -1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target)
        self.log('val_loss', loss)

        gen_images = self.transformerLM_t2i.generate_image(
            texts,
            images[:, :-img_len],
            view,
            condition_len,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        output = {
            'GT_image': images,
            'gen_image': gen_images,
            'img_paths': img_paths,
            'val_loss': loss,
            'view': view,
        }
        return output

    def validation_epoch_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(
            validation_step_outputs)

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        if self.trainer.is_global_zero:
            self.log("val_loss_epoch", total_val_loss)

        if self.global_rank == 0:
            torch.save(gathered_validation_step_outputs, os.path.join(
                self.save_dir, str(self.current_epoch) + '_eval_output.pt'))
        time.sleep(0.5)

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats
        img_paths, images, texts, view = batch['img_paths'], batch['images'], batch['texts'], batch['view_position']
        condition_len = self.kargs['condition_len']
        img_len = self.kargs['img_len']

        logit = self(images, texts, view)

        target = images[:, -img_len:].reshape(-1)
        logit = logit[:, (condition_len - 1): -1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target)
        self.log('test_loss', loss)

        gen_images = self.transformerLM_t2i.generate_image(
            texts,
            images[:, :-img_len],
            view,
            condition_len,
            beam_size=self.beam_size,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        output = {
            'GT_image': images,
            'gen_image': gen_images,
            'img_paths': img_paths,
            'test_loss': loss,
            'view': view,
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)
        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        if self.trainer.is_global_zero:
            self.log('test_loss_epoch', total_test_loss)

        if self.global_rank == 0:
            torch.save(gathered_test_step_outputs, os.path.join(
                self.save_dir, 'test_output.pt'))  # test output 결과 저장하기
        time.sleep(0.5)

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.transformerLM_t2i.image_pos_emb.named_parameters():
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
################
# EfficientNet #
################


class EfficientNetLightning_t2i(pl.LightningModule):
    def __init__(self, model, lr=5e-4, weight_decay=0.01, save_dir="", infer_name="", **kargs):
        super().__init__()
        self.kargs = kargs
        self.model = model(**kargs)
        pos_weight = torch.FloatTensor([5] * 14)
        # self.criterion = FocalLoss()
        # self.criterion = nn.MultiLabelSoftMarginLoss()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight)  # cross_entropy()
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_dir = save_dir
        self.infer_name = infer_name
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters()

    def forward(self, images):
        logit = self.model(images)
        return logit

    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']

        fw_starttime = time.monotonic()
        logit = self(images)
        fw_endtime = time.monotonic()

        # logit = logit.reshape(-1, logit.size(-1))
        # labels = labels.reshape(-1, labels.size(-1))
        loss = self.criterion(logit.to(torch.float), labels)

        '''
        # log 6 example images
        # or generated text... or whatever
        sample_imgs = x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('example_images', grid, 0)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        '''

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
            'peak_mem': _peak_mem,
        }

        return output

    def validation_step(self, batch, batch_idx):
        images, labels = batch['images'].cuda(), batch['labels'].cuda()

        fw_starttime = time.monotonic()
        logit = self(images)
        fw_endtime = time.monotonic()

        _fw_time = math.log2(fw_endtime - fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        loss = self.criterion(logit, labels)

        labels = labels.detach().cpu().numpy()  # print(labels) # b, 14
        logit = torch.sigmoid(logit).detach(
        ).cpu().numpy()  # print(logit) # b, 14

        fpr, tpr, roc_auc = dict(), dict(), dict()
        precision, recall, ap, f1 = dict(), dict(), dict(), dict()
        _, n_classes = labels.shape
        for i in range(n_classes):
            fpr[str(i)], tpr[str(i)], _ = roc_curve(labels[:, i], logit[:, i])
            # roc_auc[str(i)] = auc(fpr[str(i)], tpr[str(i)])
            try:
                roc_auc[str(i)] = roc_auc_score(labels[:, i], logit[:, i])
            except ValueError:
                roc_auc[str(i)] = 0.0

            precision[str(i)], recall[str(i)], _ = precision_recall_curve(
                labels[:, i], logit[:, i])
            ap[str(i)] = average_precision_score(labels[:, i], logit[:, i])
            f1[str(i)] = f1_score(labels[:, i],
                                  logit[:, i] > 0.5, zero_division=0)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            labels.ravel(), logit.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        try:
            roc_auc["micro"] = roc_auc_score(
                labels.ravel(), logit.ravel(), average='micro')
        except ValueError:
            roc_auc["micro"] = 0.0
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            labels.ravel(), logit.ravel())
        ap["micro"] = average_precision_score(labels, logit, average='micro')
        f1["micro"] = f1_score(labels, logit > 0.5,
                               average="micro", zero_division=0)

        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_fw_time', _fw_time, on_step=True,
                 on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('val_peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(roc_auc, on_step=False, on_epoch=True, sync_dist=True,)
        self.log_dict(ap, on_step=False, on_epoch=True, sync_dist=True,)
        self.log_dict(f1, on_step=False, on_epoch=True, sync_dist=True,)

        output = {
            'batch_idx': batch_idx,
            'val_loss': loss,
            'val_fw_time': _fw_time,
            'val_peak_mem': _peak_mem,

            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'f1': f1,
        }

        return output

    def validation_epoch_end(self, validation_step_outputs):
        # gathered_validation_step_outputs = self.all_gather(validation_step_outputs, sync_grads=True)
        gathered_validation_step_outputs = validation_step_outputs

        total_val_loss = torch.mean(
            gathered_validation_step_outputs[0]['val_loss'])
        total_val_f1 = np.mean(
            gathered_validation_step_outputs[0]['f1']['micro'])
        total_val_ap = np.mean(
            gathered_validation_step_outputs[0]['ap']['micro'])

        fpr = gathered_validation_step_outputs[0]['fpr']
        tpr = gathered_validation_step_outputs[0]['tpr']
        roc_auc = gathered_validation_step_outputs[0]['roc_auc']
        precision = gathered_validation_step_outputs[0]['precision']
        recall = gathered_validation_step_outputs[0]['recall']
        ap = gathered_validation_step_outputs[0]['ap']

        if self.trainer.is_global_zero:
            self.log("val_loss_epoch", total_val_loss)
            self.log("total_val_f1", total_val_f1)
            self.log("total_val_ap", total_val_ap)

            colors = cycle(['red', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'indigo',
                            'darkgreen', 'brown', 'crimson', 'fuchsia', 'olive', 'indianred', 'chocolate'])

            label_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
                           'Fracture', 'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                           'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

            # ! # 마이크로 평균 정밀 리콜 곡선 플롯
            plt.figure(figsize=(7, 8))
            lines, labels = [], []
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0],
                              color='gray', alpha=0.2, lw=1)
                plt.annotate('f1={0:0.1f}'.format(
                    f_score), xy=(0.9, y[45] + 0.02))

            for i, color in zip(range(14), colors):
                l, = plt.plot(recall[str(i)],
                              precision[str(i)], color=color, lw=1)
                lines.append(l)
                labels.append(
                    '({0}) Precision-recall for class {1} (area = {2:0.2f})'.format(i, label_names[i], ap[str(i)]))

            l, = plt.plot(recall['micro'],
                          precision['micro'], color='black', lw=2)
            lines.append(l)
            labels.append(
                'micro-average Precision-recall (area = {0:0.2f})'.format(ap["micro"]))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.55)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Valid Precision-Recall curve to multi-class')
            plt.legend(lines, labels, loc=(0, -1.4), prop=dict(size=9))
            plt.savefig(os.path.join(self.save_dir,
                        'eval_pr_curve_multi.png'), dpi=300)
            plt.close()

            # ! # Plot all ROC curves
            plt.figure(figsize=(7, 8))
            lines, labels = [], []
            for i, color in zip(range(14), colors):
                l, = plt.plot(fpr[str(i)], tpr[str(i)], color=color, lw=1)
                lines.append(l)
                labels.append("({0}) ROC curve of class {1} (area = {2:0.2f})".format(
                    i, label_names[i], roc_auc[str(i)]))

            l, = plt.plot(fpr["micro"], tpr["micro"],
                          color="black", linestyle=":", linewidth=3)
            lines.append(l)
            labels.append(
                "micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.55)
            plt.plot([0, 1], [0, 1], "k--", lw=1, color='gray')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Valid Receiver Operating Characteristic to multiclass")
            plt.legend(lines, labels, loc=(0, -1.4), prop=dict(size=9))
            plt.savefig(os.path.join(self.save_dir,
                        'eval_roc_curve_multi.png'), dpi=300)
            plt.close()

    def test_step(self, batch, batch_idx):
        images, labels = batch['images'].cuda(), batch['labels'].cuda()

        fw_starttime = time.monotonic()
        logit = self(images)
        fw_endtime = time.monotonic()

        _fw_time = math.log2(fw_endtime - fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        loss = self.criterion(logit, labels)

        labels = labels.detach().cpu().numpy()  # print(labels) # b, 14
        logit = torch.sigmoid(logit).detach(
        ).cpu().numpy()  # print(logit) # b, 14

        fpr, tpr, test_roc_auc = dict(), dict(), dict()
        precision, recall, test_ap, test_f1 = dict(), dict(), dict(), dict()
        _, n_classes = labels.shape
        for i in range(n_classes):
            fpr[str(i)], tpr[str(i)], _ = roc_curve(labels[:, i], logit[:, i])
            # roc_auc[str(i)] = auc(fpr[str(i)], tpr[str(i)])
            try:
                test_roc_auc[str(i)] = roc_auc_score(
                    labels[:, i], logit[:, i])  # , average='weighted')
            except ValueError:
                test_roc_auc[str(i)] = 0.0

            precision[str(i)], recall[str(i)], _ = precision_recall_curve(
                labels[:, i], logit[:, i])
            test_ap[str(i)] = average_precision_score(
                labels[:, i], logit[:, i])
            test_f1[str(i)] = f1_score(labels[:, i], logit[:, i] > 0.5)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            labels.ravel(), logit.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        try:
            test_roc_auc["micro"] = roc_auc_score(
                labels.ravel(), logit.ravel(), average='micro')
        except ValueError:
            test_roc_auc["micro"] = 0.0

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            labels.ravel(), logit.ravel())
        test_ap["micro"] = average_precision_score(
            labels, logit, average='micro')
        test_f1["micro"] = f1_score(labels, logit > 0.5, average="micro")

        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log('test_fw_time', _fw_time, on_step=True,
                 on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('test_peak_mem', _peak_mem, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(test_roc_auc, on_step=False,
                      on_epoch=True, sync_dist=True,)
        self.log_dict(test_ap, on_step=False, on_epoch=True, sync_dist=True,)
        self.log_dict(test_f1, on_step=False, on_epoch=True, sync_dist=True,)

        output = {
            'batch_idx': batch_idx,
            'test_loss': loss,
            'test_fw_time': _fw_time,
            'test_peak_mem': _peak_mem,

            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': test_roc_auc,
            'precision': precision,
            'recall': recall,
            'ap': test_ap,
            'f1': test_f1,
        }
        return output

    def test_epoch_end(self, test_step_outputs):
        # gathered_test_step_outputs = self.all_gather(test_step_outputs)
        gathered_test_step_outputs = test_step_outputs

        total_test_loss = torch.mean(
            gathered_test_step_outputs[0]['test_loss'])
        total_test_f1 = np.mean(gathered_test_step_outputs[0]['f1']['micro'])
        total_test_ap = np.mean(gathered_test_step_outputs[0]['ap']['micro'])

        fpr = gathered_test_step_outputs[0]['fpr']
        tpr = gathered_test_step_outputs[0]['tpr']
        roc_auc = gathered_test_step_outputs[0]['roc_auc']
        precision = gathered_test_step_outputs[0]['precision']
        recall = gathered_test_step_outputs[0]['recall']
        ap = gathered_test_step_outputs[0]['ap']

        if self.trainer.is_global_zero:
            self.log("test_loss_epoch", total_test_loss)
            self.log("total_test_f1", total_test_f1)
            self.log("total_test_ap", total_test_ap)

            colors = cycle(['red', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'indigo',
                            'darkgreen', 'brown', 'crimson', 'fuchsia', 'olive', 'indianred', 'chocolate'])

            label_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
                           'Fracture', 'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                           'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

            # ! # 마이크로 평균 정밀 리콜 곡선 플롯
            plt.figure(figsize=(7, 8))
            lines, labels = [], []
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0],
                              color='gray', alpha=0.2, lw=1)
                plt.annotate('f1={0:0.1f}'.format(
                    f_score), xy=(0.9, y[45] + 0.02))

            # lines.append(l)
            # labels.append('iso-f1 curves')

            for i, color in zip(range(14), colors):
                l, = plt.plot(recall[str(i)],
                              precision[str(i)], color=color, lw=1)
                lines.append(l)
                labels.append(
                    '({0}) Precision-recall for class {1} (area = {2:0.2f})'.format(i, label_names[i], ap[str(i)]))

            l, = plt.plot(recall['micro'],
                          precision['micro'], color='black', lw=2)
            lines.append(l)
            labels.append(
                'micro-average Precision-recall (area = {0:0.2f})'.format(ap["micro"]))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.55)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Test Precision-Recall curve to multi-class')
            plt.legend(lines, labels, loc=(0, -1.4), prop=dict(size=9))
            plt.savefig(os.path.join(
                self.save_dir, 'test_pr_curve_multi_' + self.infer_name + '.png'), dpi=300)
            plt.close()

            # ! # Plot all ROC curves
            plt.figure(figsize=(7, 8))
            lines, labels = [], []
            for i, color in zip(range(14), colors):
                l, = plt.plot(fpr[str(i)], tpr[str(i)], color=color, lw=1)
                lines.append(l)
                labels.append("({0}) ROC curve of class {1} (area = {2:0.2f})".format(
                    i, label_names[i], roc_auc[str(i)]))

            l, = plt.plot(fpr["micro"], tpr["micro"],
                          color="black", linestyle=":", linewidth=3)
            lines.append(l)
            labels.append(
                "micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.55)
            plt.plot([0, 1], [0, 1], "k--", lw=1, color='gray')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Test Receiver Operating Characteristic to multiclass")
            plt.legend(lines, labels, loc=(0, -1.4), prop=dict(size=9))
            plt.savefig(os.path.join(
                self.save_dir, 'test_roc_curve_multi_' + self.infer_name + '.png'), dpi=300)
            plt.close()

    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.model.named_parameters():
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


"""
class EfficientNetLightning_t2i_before(pl.LightningModule):
    def __init__(self, model, lr=5e-4, weight_decay=0.01, save_dir="", **kargs):
        super().__init__()
        self.kargs = kargs
        self.efficientnet_t2i = model
        self.criterion = nn.MultiLabelSoftMarginLoss()
        # self.criterion = nn.BCEWithLogitsLoss()   #cross_entropy(
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_dir = save_dir
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters()

    def forward(self, images, **kargs):
        logit = self.efficientnet_t2i(images, **kargs)
        return logit


    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']

        fw_starttime = time.monotonic()
        logit = self(images)
        fw_endtime = time.monotonic()

        # logit = logit.reshape(-1, logit.size(-1))
        loss = self.criterion(logit, labels)

        labels = labels.detach().cpu().numpy()  # print(labels) # b, 14
        logit = logit.detach().cpu().numpy()  # print(logit) # b, 14

        _ap_micro = average_precision_score(labels, logit, average='micro')
        # _f1 = f1_score(labels, logit>0.1, average='samples')
        # _f1_macro = f1_score(labels, logit>0.1, average="macro")
        _f1_micro = f1_score(labels, logit>0.0001, average="micro")
        _fw_time = math.log2(fw_endtime-fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('ap_micro', _ap_micro, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('f1_micro', _f1_micro, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('fw_time', _fw_time, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('peak_mem', _peak_mem, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'batch_idx': batch_idx,
            'loss': loss,
            'ap_micro':_ap_micro,
            'f1_micro':_f1_micro,
            'fw_time': _fw_time,
            'peak_mem': _peak_mem,
            }

        return output



    def validation_step(self, batch, batch_idx):
        images, labels = batch['images'].cuda(), batch['labels']

        fw_starttime = time.monotonic()
        logit = self(images)
        fw_endtime = time.monotonic()

        loss = self.criterion(logit, labels)

        labels = labels.detach().cpu().numpy()  # print(labels) # b, 14
        logit = logit.detach().cpu().numpy()  # print(logit) # b, 14

        _ap_micro = average_precision_score(labels, logit, average='micro')
        # _precision_micro = precision_score(labels, logit>0.01, average="micro")
        # _recall_micro = recall_score(labels, logit>0.01, average="micro")
        _f1_micro = f1_score(labels, logit>0.0001, average="micro")
        _fw_time = math.log2(fw_endtime-fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        _, n_classes = labels.shape
        precision_curve, recall_curve, average_precision = dict(), dict(), dict()
        for i in range(n_classes):
            precision_curve[i], recall_curve[i], _ = precision_recall_curve(labels[:, i], logit[:, i]>0.0001)
            average_precision[i] = average_precision_score(labels[:, i], logit[:, i])

        # "미시적 평균": 모든 클래스의 점수를 공동으로 정량화
        precision_curve["micro"], recall_curve["micro"], _ = precision_recall_curve(labels.ravel(), logit.ravel())
        average_precision["micro"] = average_precision_score(labels, logit, average="micro")

        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_ap_micro', _ap_micro, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_f1_micro', _f1_micro, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_fw_time', _fw_time, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('val_peak_mem', _peak_mem, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'batch_idx': batch_idx,
            'val_loss': loss,
            'val_ap_micro':_ap_micro,
            'val_f1_micro':_f1_micro,
            'val_fw_time': _fw_time,
            'val_peak_mem': _peak_mem,
            'precision_curve':precision_curve,
            'recall_curve':recall_curve,
            'average_precision':average_precision,
            }

        return output


    def validation_epoch_end(self, validation_step_outputs):
        gathered_validation_step_outputs = self.all_gather(validation_step_outputs)

        total_val_loss = torch.mean(gathered_validation_step_outputs[0]['val_loss'])
        total_val_ap = torch.mean(gathered_validation_step_outputs[0]['val_ap_micro'])
        total_val_f1 = torch.mean(gathered_validation_step_outputs[0]['val_f1_micro'])
        precision_curve = gathered_validation_step_outputs[0]['precision_curve']
        recall_curve = gathered_validation_step_outputs[0]['recall_curve']
        average_precision = gathered_validation_step_outputs[0]['average_precision']

        if self.trainer.is_global_zero:
            self.log("val_loss_epoch", total_val_loss)
            self.log("total_val_ap", total_val_ap)
            self.log("total_val_f1", total_val_f1)


            # 마이크로 평균 정밀 리콜 곡선 플롯
            plt.figure()
            plt.step(torch.mean(recall_curve['micro'], dim=0).detach().cpu().numpy(), torch.mean(precision_curve['micro'], dim=0).detach().cpu().numpy(), where='post')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Valid Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(torch.mean(average_precision["micro"]).detach().cpu().numpy()))
            plt.savefig(os.path.join(self.save_dir, 'eval_ap_curve.png'), dpi=300)


            # 설정 플롯 세부 사항
            colors = cycle(['red', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'indigo',
                            'darkgreen', 'brown', 'crimson', 'fuchsia', 'olive', 'indianred', 'chocolate'])

            label_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
                           'Fracture', 'Lung_Lesion', 'Lung_Opacity','No_Finding', 'Pleural_Effusion',
                           'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

            plt.figure(figsize=(7, 8))
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            labels = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

            lines.append(l)
            labels.append('iso-f1 curves')
            l, = plt.plot(torch.mean(recall_curve['micro'], dim=0).detach().cpu().numpy(), torch.mean(precision_curve['micro'], dim=0).detach().cpu().numpy(), color='gold', lw=2)
            lines.append(l)
            labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(torch.mean(average_precision["micro"]).detach().cpu().numpy()))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.35)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Valid Precision-Recall curve')
            plt.legend(lines, labels, loc=(0, -0.3), prop=dict(size=9))
            plt.savefig(os.path.join(self.save_dir, 'eval_pr_curve.png'), dpi=300)



            # for i, color in zip(range(14), colors):
            #     l, = plt.plot(torch.mean(recall_curve[i], dim=0).detach().cpu().numpy(), torch.mean(precision_curve[i], dim=0).detach().cpu().numpy(), color=color, lw=2)
            #     lines.append(l)
            #     labels.append('({0}) Precision-recall for class {1} (area = {2:0.2f})'
            #                 ''.format(i, label_names[i], torch.mean(average_precision[i]).detach().cpu().numpy()))

            # fig = plt.gcf()
            # fig.subplots_adjust(bottom=0.55)
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('Valid Precision-Recall curve to multi-class')
            # plt.legend(lines, labels, loc=(0, -1.4), prop=dict(size=9))
            # plt.savefig(os.path.join(self.save_dir, 'eval_pr_curve_multi.png'), dpi=300)
            # plt.close()




    def test_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']

        fw_starttime = time.monotonic()
        logit = self(images)  # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        fw_endtime = time.monotonic()

        # logit = logit.reshape(-1, logit.size(-1))
        loss = self.criterion(logit, labels)

        labels = labels.detach().cpu().numpy()  # print(labels) # b, 14
        logit = logit.detach().cpu().numpy()  # print(logit) # b, 14

        _ap_micro = average_precision_score(labels, logit, average='micro')
        # _precision_micro = precision_score(labels, logit>0.01, average="micro")
        # _recall_micro = recall_score(labels, logit>0.01, average="micro")
        _f1_micro = f1_score(labels, logit>0.0001, average="micro")
        _fw_time = math.log2(fw_endtime-fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        _, n_classes = labels.shape
        precision_curve, recall_curve, average_precision = dict(), dict(), dict()
        for i in range(n_classes):
            precision_curve[i], recall_curve[i], _ = precision_recall_curve(labels[:, i], logit[:, i]>0.0001)
            average_precision[i] = average_precision_score(labels[:, i], logit[:, i])

        # "미시적 평균": 모든 클래스의 점수를 공동으로 정량화
        precision_curve["micro"], recall_curve["micro"], _ = precision_recall_curve(labels.ravel(), logit.ravel())
        average_precision["micro"] = average_precision_score(labels, logit, average="micro")


        _fw_time = math.log2(fw_endtime-fw_starttime)
        _peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('fw_time', _fw_time, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('peak_mem', _peak_mem, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('test_ap_micro', _ap_micro, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('test_f1_micro', _f1_micro, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        output = {
            'batch_idx': batch_idx,
            'test_loss': loss,
            'test_ap_micro':_ap_micro,
            'test_f1_micro':_f1_micro,
            'test_fw_time': _fw_time,
            'test_peak_mem': _peak_mem,
            'precision_curve':precision_curve,
            'recall_curve':recall_curve,
            'average_precision':average_precision,
            }
        return output


    def test_epoch_end(self, test_step_outputs):
        gathered_test_step_outputs = self.all_gather(test_step_outputs)
        total_test_loss = torch.mean(gathered_test_step_outputs[0]['test_loss'])
        total_test_ap = torch.mean(gathered_test_step_outputs[0]['test_ap_micro'])
        total_test_f1 = torch.mean(gathered_test_step_outputs[0]['test_f1_micro'])
        precision_curve = gathered_test_step_outputs[0]['precision_curve']
        recall_curve = gathered_test_step_outputs[0]['recall_curve']
        average_precision = gathered_test_step_outputs[0]['average_precision']

        if self.trainer.is_global_zero:
            self.log("test_loss_epoch", total_test_loss)
            self.log("total_test_ap", total_test_ap)
            self.log("total_test_f1", total_test_f1)
            self.log_dict("total_test_f1", average_precision)

            # 마이크로 평균 정밀 리콜 곡선 플롯
            plt.figure()
            plt.step(torch.mean(recall_curve['micro'], dim=0).detach().cpu().numpy(), torch.mean(precision_curve['micro'], dim=0).detach().cpu().numpy(), where='post')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Test Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(torch.mean(average_precision["micro"]).detach().cpu().numpy()))
            plt.savefig(os.path.join(self.save_dir, 'test_ap_curve.png'), dpi=300)
            plt.close()

            # 설정 플롯 세부 사항
            colors = cycle(['red', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'indigo',
                            'darkgreen', 'brown', 'crimson', 'fuchsia', 'olive', 'indianred', 'chocolate'])

            label_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
                           'Fracture', 'Lung_Lesion', 'Lung_Opacity','No_Finding', 'Pleural_Effusion',
                           'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

            plt.figure(figsize=(7, 8))
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            labels = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

            lines.append(l)
            labels.append('iso-f1 curves')
            l, = plt.plot(torch.mean(recall_curve['micro'], dim=0).detach().cpu().numpy(), torch.mean(precision_curve['micro'], dim=0).detach().cpu().numpy(), color='gold', lw=2)
            lines.append(l)
            labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(torch.mean(average_precision["micro"]).detach().cpu().numpy()))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.35)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Test Precision-Recall curve')
            plt.legend(lines, labels, loc=(0, -0.3), prop=dict(size=9))
            plt.savefig(os.path.join(self.save_dir, 'test_pr_curve.png'), dpi=300)
            plt.close()

            for i, color in zip(range(14), colors):
                l, = plt.plot(torch.mean(recall_curve[i], dim=0).detach().cpu().numpy(), torch.mean(precision_curve[i], dim=0).detach().cpu().numpy(), color=color, lw=2)
                lines.append(l)
                labels.append('({0}) Precision-recall for class {1} (area = {2:0.2f})'
                            ''.format(i, label_names[i], torch.mean(average_precision[i]).detach().cpu().numpy()))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.55)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Test Precision-Recall curve to multi-class')
            plt.legend(lines, labels, loc=(0, -1.4), prop=dict(size=9))
            plt.savefig(os.path.join(self.save_dir, 'test_pr_curve.png'), dpi=300)
            plt.close()



    def configure_optimizers(self):

        all_params = set(self.parameters())
        wd_params = set()
        decay_module = (nn.Embedding, nn.Linear, nn.Conv2d)
        for m in self.modules():
            if isinstance(m, decay_module):
                wd_params.add(m.weight)

        # manually add
        for n, p in self.efficientnet_t2i.named_parameters():
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



"""
