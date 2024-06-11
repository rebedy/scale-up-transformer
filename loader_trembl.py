from torch.utils.data import Dataset
import torch
import numpy as np

# This post-processor takes care of adding the special tokens: a [EOS] token and a [SOS] token
import random
random.seed(42)


class TREMBLDataset(Dataset):

    def __init__(self,
                 file_path,
                 max_seq_len,
                 pad_idx,
                 ):
        super().__init__()

        self.sequence = np.load(file_path, allow_pickle=True)[:100000]
        self.pad_idx = 3

    def make_pad_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        # trg_sub_mask = [trg len, trg len]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()
        # trg_mask = [batch size, 1, trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def get_attn_decoder_mask(seq):
        subsequent_mask = torch.ones_like(
            seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
        # upper triangular part of a matrix(2-D)
        subsequent_mask = subsequent_mask.triu(diagonal=1)
        return subsequent_mask

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, i):

        seq = torch.Tensor(self.sequence[i])

        mask = self.make_pad_mask(seq)

        return {
            'seq': seq,
            'mask': mask,
        }
