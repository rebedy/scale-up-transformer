import os

import csv
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from PIL import Image
import albumentations
import albumentations.pytorch

import torch
torch.set_printoptions(profile="full")

from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing 
# This post-processor takes care of adding the special tokens: a [EOS] token and a [SOS] token
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import json
from vae import VQGanVAE

from utils import *
from helpers import *
random_seed_all(42)


def _pad_sequence(sequences, max_len, padding_value=0):
    trailing_dims = sequences[0].size()[1:]
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor
    return out_tensor

#! jh

class CXRDataset(Dataset):

    def __init__(self,
                metadata_file,
                img_root_dir, 
                text_root_dir,
                vqgan_model_path,
                vqgan_config_path,
                codebook_indices_path,
                max_img_num,   # eg. 4
                max_text_len,  # eg. 512
                img_resol,
                vocab_file,
                merge_file,
                target_count,
                under_sample
                ):
        super(CXRDataset,self).__init__()
        
        self.img_root_dir = img_root_dir    #input
        self.max_img_num = max_img_num
        self.text_root_dir = text_root_dir  #target
        self.max_text_len = max_text_len
        self.under_sample = under_sample
        
        
        ## Read CSV
        self.dict_by_studyid = defaultdict(list)
        f = open(metadata_file[1], 'r')
        rdr = csv.reader(f)
        for i, line in enumerate(tqdm(rdr)):
            dicom_id, subject_id, study_id, ViewPosition, count = line
            if (int(count) == int(target_count)):
                self.dict_by_studyid[study_id].append(line)  # {study_id: [[dicom_id, subject_id, study_id, ViewPosition, count],[...],...]}
        self.key_list = list(self.dict_by_studyid.keys())
        print("\nnumber of target subject:", len(self.key_list))
        
        data_path = metadata_file[0]
        self.data = [json.loads(l) for l in open(data_path)] #studyid , matching이 되면? image& text

        ## Tokenizer
        self.tokenizer = ByteLevelBPETokenizer(
            vocab_file,
            merge_file,
        )  #161,221,118
        self.tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
        )
        self.tokenizer.enable_truncation(max_length=max_text_len)  # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=max_text_len)  # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다
        self.pad_token_idx = self.tokenizer.token_to_id("[PAD]")
        self.text_vocab_size = self.tokenizer.get_vocab_size()

        # #!# VAE params and load cookbook -> NO 필요!!!
        self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)
        self.img_fmap_size = self.vae.fmap_size
        self.img_vocab_size = self.vae.num_tokens  # eg. 1024
        
        # self.img_reso = self.vae.image_size        # eg. 256 or 384 in my case       
        self.img_reso = img_resol  
        # self.img_len = int((self.img_reso / self.vae.f)**2)
        ## eg. 16**2 = 256  -> 512를 넣었을때
        ## eg. 8**2 = 64  -> 256을 넣었을때
        if img_resol == 512:
            self.img_len = 16**2
        elif img_resol == 256:
            self.img_len = 8**2
        else:
            print("[WARN] Please check the image resolution.")
            self.img_len = 256
            pass
        
        ## Image
        self.rescaler = albumentations.SmallestMaxSize(max_size = img_resol)    # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.cropper = albumentations.CenterCrop(height=img_resol, width=img_resol)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        
    def __len__(self):
        return len(self.data)
    
    def preprocess_image(self, image_path):  # not used now
        image = Image.open(image_path)   # PIL format
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)         # value 0 ~ 255
        image = self.preprocessor(image=image)["image"]  # albumentations의 output: {'image': numpy.ndarray(256,256,3)}
        image = (image/255.0).astype(np.float32)         # Note that you have to make image in value 0. ~ 1.
        return torch.tensor(image)   # ndarray (256, 256, 3)

    def __getitem__(self, i):
        d_study_id, _, label, txt, img = self.data[i].keys()  # id, txt, img
        d_study_id = self.data[i][d_study_id]
        d_label = self.data[i][label]
        d_txt = self.data[i][txt]
        d_img = self.data[i][img]
        input_img = self.preprocess_image(d_img)  #torch.Size([512, 512, 3])
        # study_id = self.key_list[i]
        
        "요기가 이미지 들어가는 부분"
        ### image
        # img_cls,_ = self.cnn(input_img)
        
        # if len(self.dict_by_studyid[study_id]) > self.max_img_num: #if there are 2 slot but got 1 image,
        #     ## fixed1of2
        #     if self.under_sample == 'fixed':
        #         imgs_meta = [self.dict_by_studyid[study_id][0]]
        #         count = 1
        #     ## random1of2
        #     elif self.under_sample == 'random':
        #         imgs_meta = random.sample(self.dict_by_studyid[study_id], self.max_img_num)
        #         count = 1
        # else:
        #     ## all2of2
        #     imgs_meta =self.dict_by_studyid[study_id]
        #     count = self.max_img_num
            
        image_indices = 256
        # image_output = torch.tensor(self.slots)  # tensor[img_len * max_img_num] 2*256
        # img_paths = ''
        # for c in range(count):
        #     dicom_id, subject_id, studyid, ViewPosition, cnt = imgs_meta[c]
        #     img_path = os.path.join(self.img_root_dir, 'p'+subject_id[:2], 'p'+subject_id, 's'+studyid, dicom_id+'.jpg')
        #     image_indices = self.indices_dict[dicom_id] # indices list
        #     image_indices = torch.tensor(image_indices) # [img_len]
        #     image_output[self.img_len*c:self.img_len*(c+1)] = image_indices
        #     img_paths += (img_path + '|')
            
        #~₩₩₩₩₩₩₩₩₩₩~~~~~~~~~~~~~~~~~~~        
        ### text
        # text_path = os.path.join(self.text_root_dir, d_study_id+'.txt')
        # with open(text_path, 'r') as f:
        #     data = f.read()
        # src = data.replace('  ', ' ').replace('  ', ' ').lower()   # Note: 토크나이저가 lower text에서 학습됐음

        ids_list = self.tokenizer.encode(d_txt).ids  # len: max_text_len
        text_output = torch.tensor(ids_list)  # tensor[max_text_len]
        ### mask
        # img_mask = torch.full_like(image_output,  1.)
        # img_mask = img_mask.float().masked_fill(img_mask == 1, float('-inf')).masked_fill(img_mask == 0, float(0.0))
        # target_mask = self.make_std_mask(text_output, self.pad_token_idx)
        # target_mask = target_mask.float().masked_fill(target_mask == 1, float('-inf')).masked_fill(target_mask == 0, float(0.0))
        # img_mask = torch.full((image_output.shape[0], image_output.shape[0]), True)
        # target_mask = self.make_std_mask(text_output, self.pad_token_idx)
        """
        torch.Size([256])
        torch.Size([256])
        torch.Size([256, 256])
        torch.Size([256, 256])
        """
        # print(image_output)
        # print(text_output)
        # print(img_mask[1,...])
        # print(target_mask[1,...])
        
        return {
            'images':input_img,      # input b, 512, 512, 3
            # 'image_mask': img_mask,     # input_mask 512
            'texts': text_output,       # target
            # 'text_mask': target_mask,   # target_mask
            # 'attn_mask': attn_mask,   # target_mask
            'study_id': d_study_id, 
            'img_paths': d_img,
            'token_num': (text_output[...,1:] != self.pad_token_idx).data.sum()  # token_num
            }
    
