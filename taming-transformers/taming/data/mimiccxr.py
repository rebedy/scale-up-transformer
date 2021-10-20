import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths


class MimicCXRTrain(Dataset):

    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        root = '/home/edlab/wcshin/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        with open("data/mimiccxrtrain.txt", "r") as f:
            relpaths = f.read().splitlines()  # ['00982.png', '00988.png',...]
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)  # self.data: <class 'taming.data.base.ImagePaths'>    # size = 256
        self.keys = keys  # 보통 None

        self.coord = coord   # False
        if crop_size is not None:  # crop_size = 256
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]  # self.data: <taming.data.base.ConcatDatasetWithIndex object>  ex = {image': array(256, 256, 3), 'file_path_': 'data/ffhq/00137.png'}
        if hasattr(self, "cropper"):
            if not self.coord:  # self.coord는 False
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        return ex


class MimicCXRValidation(Dataset):

    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        root = '/home/edlab/wcshin/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        with open("data/mimiccxrvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()  # ['00982.png', '00988.png',...]
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)  # self.data: <class 'taming.data.base.ImagePaths'>    # size = 256
        self.keys = keys  # 보통 None

        self.coord = coord  # False
        if crop_size is not None:  # crop_size = 256
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, i):
        ex = self.data[i]  # self.data: <taming.data.base.ConcatDatasetWithIndex object>   ex = {image': array(256, 256, 3), 'file_path_': 'data/ffhq/00137.png'}
        if hasattr(self, "cropper"): # True
            if not self.coord:  # self.coord = False
                out = self.cropper(image=ex["image"])  # 이미 ex["image"]가 256x256으로 centercrop된 후에 나온건데 또 centercrop을 하네.
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        return ex