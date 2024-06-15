import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from performer_pytorch import PerformerLM_i2t


class PerformerLightning_i2t(pl.LightningModule):
    def __init__(self, lr=5e-4, ignore_index=0, **kargs):
        super().__init__()
        self.kargs = kargs
        self.performerLM_i2t = PerformerLM_i2t(**kargs)
        self.lr = lr
        self.ignore_index = ignore_index
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters()

    def forward(self, images, texts):
        logit = self.performerLM_i2t(images, texts)
        return logit

    # batch: {'images': tensor[B, img_len * max_img_num], 'texts': tensor[B, max_text_len]}
    def training_step(self, batch, batch_idx):
        images, texts = batch['images'], batch['texts']
        # -> [B, img_len * max_img_num + max_text_len, num_tokens]  # NOTE: num_tokens = text_vocab_size
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.ignore_index)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch['images'], batch['texts']
        logit = self(images, texts)
        condition_len = self.kargs['condition_len']
        target = texts[:, 1:].reshape(-1)
        logit = logit[:, condition_len:-1].reshape(-1, logit.size(-1))
        loss = cross_entropy(logit, target, ignore_index=self.ignore_index)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
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
