import torch
from torch.utils.data import Dataset

import numpy as np
import os
import pickle

class TremblDataset(Dataset):
    def __init__(self,
                max_input_len,
                train_type
                ):
        super().__init__()

        assert train_type in ['train', 'valid', 'test']

        # load dataset
        self.dataset = np.load(os.path.join('/home/data_storage/dylee_135/trembl', f'trembl_{max_input_len:04d}', f'{train_type}_{max_input_len}.npy'))

        # load vocabulary
        # with open('/home/data_storage/dylee_135/trembl/vocab.pkl', 'rb') as f:
        #     self.vocab = pickle.load(f)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        data = self.dataset[item]
        return torch.tensor(data).long()

        