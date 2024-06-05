import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CXRDataModule(pl.LightningDataModule):
    def __init__(self,
                 train=None,
                 val=None,
                 test=None,
                 batch_size=16,
                 num_workers=0,
                 ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=False)
