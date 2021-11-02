import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]      # eg. example = { "image": ndarray(256, 256, 3) value -1.0 ~ 1.0  ,  "file_path_": 'data/ffhq/00141.png' }   from ImagePaths's __getitem__
        ex = {}
        if self.keys is not None:   # 보통 self.keys = None
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex  # 보통 { "image": ndarray(256, 256, 3) value -1.0 ~ 1.0  ,  "file_path_": 'data/ffhq/00141.png' }


class CelebAHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):  # size = 256, keys = None
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain_sub.txt", "r") as f:
            relpaths = f.read().splitlines()  # ['00982.png', '00988.png',...]
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)  # self.data: <class 'taming.data.base.ImagePaths'>    # size = 256
        self.keys = keys  # 보통 None


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation_sub.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys  # 보통 None


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        #d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)   # d2: <taming.data.faceshq.FFHQTrain object>
        self.data = ConcatDatasetWithIndex([d2])  # original code: [d1, d2]
        self.coord = coord   # False
        if crop_size is not None:  # crop_size = 256
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]  # self.data: <taming.data.base.ConcatDatasetWithIndex object>
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
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        #d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d2])  # original code: [d1, d2]
        self.coord = coord  # False
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]  # self.data: <taming.data.base.ConcatDatasetWithIndex object>   ex = {image': array(256, 256, 3), 'file_path_': 'data/ffhq/00137.png'}  y = 0
        if hasattr(self, "cropper"): # True
            if not self.coord:  # self.coord = False
                out = self.cropper(image=ex["image"])  # 이미 ex["image"]가 256x256으로 centercrop된 후에 나온건데 또 centercrop을 하네. 중복.
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


# FacesHQTrain, FacesHQValidation 간소화 버전
"""
class FacesHQTrain(Dataset):

    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain_sub.txt", "r") as f:
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


class FacesHQValidation(Dataset):

    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation_sub.txt", "r") as f:
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
"""