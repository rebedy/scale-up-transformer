import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import csv
import pickle
from collections import defaultdict
from tqdm import tqdm
import albumentations
import albumentations.pytorch
from vae import VQGanVAE
import random
random.seed(42)


class UnifiedCXRDataset(Dataset):

    def __init__(self,
                 metadata_file,
                 img_root_dir,  # eg. '/home/edlab/dylee/mimic/physionet.org/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
                 text_root_dir,  # eg. '/home/edlab/dylee/mimic/physionet.org/physionet.org/files/mimic-cxr-jpg/2.0.0/preprocessed_reports'
                 vqgan_model_path,
                 vqgan_config_path,
                 codebook_indices_path,
                 vqgan,
                 max_img_num,   # eg. 4
                 max_text_len,  # eg. 512
                 tokenizer,
                 target_count,  # number of images
                 target_view,   # list
                 under_sample="fixed"
                 ):
        super().__init__()
        self.dict_by_studyid = defaultdict(list)
        f = open(metadata_file, 'r')
        rdr = csv.reader(f)
        for i, line in tqdm(enumerate(rdr)):
            dicom_id, subject_id, study_id, ViewPosition, count = line
            if (int(count) == int(target_count) and ViewPosition in target_view):
                # {study_id: [[dicom_id, subject_id, study_id, ViewPosition, count],[...],...]}
                self.dict_by_studyid[study_id].append(line)
        self.key_list = list(self.dict_by_studyid.keys())  # [:16]
        print("number of target subject:", len(self.key_list))

        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir

        self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

        # when Q resol == 256
        if vqgan == 256:
            self.img_fmap_size = 16
            self.img_reso = 256        # eg. 256 or 512 in my case
            self.img_len = 256 + 2  # eg. 16**2 = 256
            self.img_vocab_size = self.vae.num_tokens  # eg. 1024
        else:
            self.img_fmap_size = 32
            self.img_reso = 512        # eg. 256 or 512 in my case
            self.img_len = 1024 + 2  # eg. 32**2 = 1024
            self.img_vocab_size = self.vae.num_tokens  # eg. 1024

        with open(codebook_indices_path, 'rb') as f:
            self.indices_dict = pickle.load(f)

        self.max_img_num = max_img_num
        self.max_text_len = max_text_len
        self.under_sample = under_sample
        self.tokenizer = tokenizer

        self.text_vocab_size = self.tokenizer.get_vocab_size()

        # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
        self.cropper = albumentations.CenterCrop(
            height=self.img_reso, width=self.img_reso)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([
            self.rescaler,
            self.cropper,
            # self.totensor,
        ])

        self.slots = []
        for i in range(self.max_img_num):
            y = [self.img_vocab_size + i] * (self.img_len)
            self.slots.extend(y)

        self.modes = ['img1', 'img2', 'txt']

    def preprocess_image(self, image_path):  # not used now
        image = Image.open(image_path)   # PIL format
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)         # value 0 ~ 255
        # albumentations의 output: {'image': numpy.ndarray(256,256,3)}
        image = self.preprocessor(image=image)["image"]
        # Note that you have to make image in value 0. ~ 1.
        image = (image / 255.0).astype(np.float32)
        return image   # ndarray (256, 256, 3)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, i):
        random.shuffle(self.modes)

        study_id = self.key_list[i]

        # ! # image
        # image meta
        # if there are 2 slot but got 1 image,
        if len(self.dict_by_studyid[study_id]) > self.max_img_num:
            # fixed1of2
            if self.under_sample == 'fixed':
                imgs_meta = [self.dict_by_studyid[study_id][0]]
                count = 1
            # random1of2
            elif self.under_sample == 'random':
                imgs_meta = random.sample(
                    self.dict_by_studyid[study_id], self.max_img_num)
                count = 1
        else:
            # all2of2
            imgs_meta = self.dict_by_studyid[study_id]
            print(imgs_meta)
            count = self.max_img_num

        # image output
        # tensor[img_len * max_img_num]
        image_output = torch.tensor(self.slots)
        img_paths = ''
        view_position = []
        for i in range(count):
            dicom_id, subject_id, studyid, ViewPosition, _ = imgs_meta[i]
            # print(imgs_meta[i])
            img_path = os.path.join(
                self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id, 's' + studyid, dicom_id + '.jpg')
            image_indices = self.indices_dict[dicom_id].copy()  # indices list
            if ViewPosition == 'AP':
                # self.tokenizer.token_to_id("[SOS1]") 1025
                image_indices.insert(0, 1025)
                # self.tokenizer.token_to_id("[EOS1]"
                image_indices.append(1026)
                image_output[self.img_len * i:self.img_len * (i + 1)] = torch.tensor(image_indices)  # [img_len]
            elif ViewPosition == 'PA':
                # self.tokenizer.token_to_id("[SOS2]")
                image_indices.insert(0, 1027)
                # self.tokenizer.token_to_id("[EOS2]")
                image_indices.append(1028)
                image_output[self.img_len * i:self.img_len * (i + 1)] = torch.tensor(image_indices)  # [img_len]
            elif ViewPosition == 'LATERAL':
                # self.tokenizer.token_to_id("[SOS3]")
                image_indices.insert(0, 1029)
                # self.tokenizer.token_to_id("[EOS3]")
                image_indices.append(1030)
                image_output[self.img_len * i:self.img_len * (i + 1)] = torch.tensor(image_indices)  # [img_len]
            elif ViewPosition == 'LL':
                # self.tokenizer.token_to_id("[SOS4]")
                image_indices.insert(0, 1029)
                # self.tokenizer.token_to_id("[EOS4]")
                image_indices.append(1030)
                image_output[self.img_len * i:self.img_len * (i + 1)] = torch.tensor(image_indices)  # [img_len]
            else:
                raise ValueError
            img_paths += (img_path + '|')
            view_position.append(ViewPosition)
        img1 = image_output[:self.img_len]
        img2 = image_output[self.img_len:self.img_len * 2]

        # ! # text
        text_path = os.path.join(self.text_root_dir, 's' + study_id + '.txt')
        with open(text_path, 'r') as f:
            data = f.read()
        src = data.replace('  ', ' ').replace(
            '  ', ' ').lower()   # Note: 토크나이저가 lower text에서 학습됐음
        ids_list = self.tokenizer.encode(src).ids  # len: max_text_len
        text_output = torch.tensor(ids_list)  # tensor[max_text_len]

        return {
            'img1': img1,
            'img2': img2,
            'txt': text_output,
            'modes': self.modes,
            'study_id': study_id,
            'img_paths': img_paths,
            'view_position': view_position,
        }


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


class UnifiedCXRDataset2(Dataset):

    def __init__(self,
                 metadata_file,
                 img_root_dir,  # eg. '/home/edlab/dylee/mimic/physionet.org/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
                 text_root_dir,  # eg. '/home/edlab/dylee/mimic/physionet.org/physionet.org/files/mimic-cxr-jpg/2.0.0/preprocessed_reports'
                 vqgan_model_path,
                 vqgan_config_path,
                 codebook_indices_path,
                 vqgan,
                 max_img_num,   # eg. 4
                 max_text_len,  # eg. 512
                 tokenizer,
                 target_count,  # number of images
                 target_view,   # list
                 under_sample="fixed"
                 ):
        super().__init__()
        self.dict_by_studyid = defaultdict(list)
        f = open(metadata_file, 'r')
        rdr = csv.reader(f)
        for i, line in tqdm(enumerate(rdr)):
            dicom_id, subject_id, study_id, ViewPosition, count = line
            if (int(count) == int(target_count) and ViewPosition in target_view):
                # {study_id: [[dicom_id, subject_id, study_id, ViewPosition, count],[...],...]}
                self.dict_by_studyid[study_id].append(line)
        self.key_list = list(self.dict_by_studyid.keys())  # [:64]
        print("number of target subject:", len(self.key_list))

        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir

        self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

        # when Q resol == 256
        if vqgan == 256:
            self.img_fmap_size = 16
            self.img_reso = 256        # eg. 256 or 512 in my case
            self.img_len = 256  # eg. 16**2 = 256
            self.img_vocab_size = self.vae.num_tokens  # eg. 1024
        else:
            self.img_fmap_size = 32
            self.img_reso = 512        # eg. 256 or 512 in my case
            self.img_len = 1024  # eg. 32**2 = 1024
            self.img_vocab_size = self.vae.num_tokens  # eg. 1024

        with open(codebook_indices_path, 'rb') as f:
            self.indices_dict = pickle.load(f)

        self.max_img_num = max_img_num
        self.max_text_len = max_text_len
        self.under_sample = under_sample
        self.tokenizer = tokenizer

        self.text_vocab_size = self.tokenizer.get_vocab_size()

        # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
        self.cropper = albumentations.CenterCrop(
            height=self.img_reso, width=self.img_reso)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([
            self.rescaler,
            self.cropper,
            # self.totensor,
        ])

        self.slots = []
        for i in range(self.max_img_num):
            y = [self.img_vocab_size + i] * (self.img_len)
            self.slots.extend(y)

        self.modes = ['img1', 'img2', 'txt']

    def preprocess_image(self, image_path):  # not used now
        image = Image.open(image_path)   # PIL format
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)         # value 0 ~ 255
        # albumentations의 output: {'image': numpy.ndarray(256,256,3)}
        image = self.preprocessor(image=image)["image"]
        # Note that you have to make image in value 0. ~ 1.
        image = (image / 255.0).astype(np.float32)
        return image   # ndarray (256, 256, 3)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, i):
        random.shuffle(self.modes)

        study_id = self.key_list[i]

        # ! # image
        # image meta
        # if there are 2 slot but got 1 image,
        if len(self.dict_by_studyid[study_id]) > self.max_img_num:
            # fixed1of2
            if self.under_sample == 'fixed':
                imgs_meta = [self.dict_by_studyid[study_id][0]]
                count = 1
            # random1of2
            elif self.under_sample == 'random':
                imgs_meta = random.sample(
                    self.dict_by_studyid[study_id], self.max_img_num)
                count = 1
        else:
            # all2of2
            imgs_meta = self.dict_by_studyid[study_id]
            count = self.max_img_num

        # image output
        # tensor[img_len * max_img_num]
        image_output = torch.tensor(self.slots)
        img_paths = ''
        view_position = []
        for i in range(count):
            dicom_id, subject_id, studyid, ViewPosition, _ = imgs_meta[i]
            img_path = os.path.join(
                self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id, 's' + studyid, dicom_id + '.jpg')
            image_indices = self.indices_dict[dicom_id]  # indices list
            image_indices = torch.tensor(image_indices)  # [img_len]
            image_output[self.img_len * i:self.img_len * (i + 1)] = image_indices
            img_paths += (img_path + '|')
            view_position.append(ViewPosition)
        img1 = image_output[:self.img_len]
        img2 = image_output[self.img_len:self.img_len * 2]

        # ! # text
        text_path = os.path.join(self.text_root_dir, 's' + study_id + '.txt')
        with open(text_path, 'r') as f:
            data = f.read()
        src = data.replace('  ', ' ').replace(
            '  ', ' ').lower()   # Note: 토크나이저가 lower text에서 학습됐음
        ids_list = self.tokenizer.encode(src).ids  # len: max_text_len
        text_output = torch.tensor(ids_list)  # tensor[max_text_len]

        return {
            'img1': img1,
            'img2': img2,
            'txt': text_output,
            'modes': self.modes,
            'study_id': study_id,
            'img_paths': img_paths,
            'view_position': view_position,
        }
