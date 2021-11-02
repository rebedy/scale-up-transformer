import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self,
                 max_input_len  # max sequence length
                 ):
        super().__init__()

        self.max_input_len = max_input_len

    def __len__(self):
        return 512

    def __getitem__(self, item):
        return torch.ones(self.max_input_len).long()
