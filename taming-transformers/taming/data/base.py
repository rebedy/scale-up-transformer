import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx    # self.datasets: [<taming.data.faceshq.FFHQValidation object>]


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):  # 보통 paths=['data/ffhq/00982.png', 'data/ffhq/00988.png',...]
        self.size = size   # 보통 256
        self.random_crop = random_crop  # 보통 False

        self.labels = dict() if labels is None else labels  # 보통 labels = None
        self.labels["file_path_"] = paths   # ['data/ffhq/00982.png', 'data/ffhq/00988.png',...]
        self._length = len(paths)

        if self.size is not None and self.size > 0:   # 보통 self.size = 256
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)    # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
            if not self.random_crop:    # 보통 self.random_crop = False
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)   # PIL format
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)         # value 0 ~ 255
        image = self.preprocessor(image=image)["image"]  # albumentations의 output: {'image': numpy.ndarray(256,256,3)}
        image = (image/127.5 - 1.0).astype(np.float32)   # value -1.0 ~ 1.0
        return image   # ndarray (256, 256, 3)

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])  # example["image"] = ndarray(256, 256, 3) value: -1.0 ~ 1.0
        for k in self.labels:
            example[k] = self.labels[k][i]   # example["file_path_"] = 'data/ffhq/00141.png'
        return example  # { "image": ndarray(256, 256, 3) value -1.0 ~ 1.0  ,  "file_path_": 'data/ffhq/00141.png' }


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
