import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer
import os

class OneBillionWordsDataset(Dataset):
    def __init__(self,
                 max_input_len,
                 train_type
                 ):
        super().__init__()
        assert train_type in ['train', 'valid', 'test']
        self.max_input_len = max_input_len

        # load dataset
        self.data = self.load_dataset(train_type)

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        inputs = self.tokenizer(sample, return_tensors='pt', padding='max_length', max_length=self.max_input_len, truncation=True)
        inputs, label = self.masking(inputs)
        return {
            'data': inputs.input_ids[0].long(),
            'label': label[0].long()
        }


    def load_dataset(self, train_type):
        data = list()
        if train_type == 'train':
            PATH = './training-monolingual.tokenized.shuffled'
            file_list = os.listdir(PATH)

            for file in file_list:
                with open(os.path.join(PATH, file), 'r') as f:
                    temp_data = f.readlines()
                    data.extend(temp_data)

        else:   # valid or test
            PATH = './heldout-monolingual.tokenized.shuffled'
            file_list = sorted(os.listdir(PATH))

            if train_type == 'valid':
                file_list = file_list[:25]
            elif train_type == 'test':
                file_list = file_list[25:]

            for file in file_list:
                with open(os.path.join(PATH, file), 'r') as f:
                    temp_data = f.readlines()
                    data.extend(temp_data)

        return data   # list of sentences

    def masking(self, inputs):
        temp_label = inputs.input_ids.detach().clone()

        # create masking
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)   # CLS, SEP, PAD는 제외하겠다

        # selection
        selection = []
        selection.append(torch.flatten(mask_arr[0].nonzero()).tolist())
        # create label
        label = torch.ones(inputs.input_ids.shape) * -100
        label[0, selection[0]] = inputs.input_ids[0, selection[0]].float()
        # masking input
        inputs.input_ids[0, selection[0]] = 103

        return (inputs, label)