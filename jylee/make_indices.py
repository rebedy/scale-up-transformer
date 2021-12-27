import os
import pdb
from vae import VQGanVAE
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

VQGAN_MODEL_PATH = '/home/edlab/jylee/Scaleup/mimiccxr_vqgan1024_res512/checkpoints/last.ckpt'
VQGAN_CONFIG_PATH = '/home/edlab/jylee/Scaleup/mimiccxr_vqgan1024_res512/configs/2021-12-17T08-58-54-project.yaml'

vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH).cuda()

import albumentations
import albumentations.pytorch
from PIL import Image
import numpy as np

IMG_SIZE = vae.image_size

rescaler = albumentations.SmallestMaxSize(max_size=IMG_SIZE)
cropper = albumentations.CenterCrop(height=IMG_SIZE, width=IMG_SIZE)
totensor = albumentations.pytorch.transforms.ToTensorV2()
image_transform = albumentations.Compose([
    rescaler,
    cropper,
    totensor
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    if not image.mode == 'RGB':
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image / 255.0).astype(np.float32)  # 0. ~ 1.
    return image

print('before image list')
from pathlib import Path
root = Path('/home/edlab/wcshin/physionet.org/files/mimic-cxr-jpg/2.0.0/files')
img_paths = [*root.glob('**/*.jpg')]
print('after image list')

import pickle
from tqdm import tqdm

dict_by_dicomid = {}

for path in tqdm(img_paths):
    dicom_id = path.stem
    image = preprocess_image(path)  # ndarray 0. ~ 1.
    image_tensor = image_transform(image=image)["image"]   # [3, 512, 512]  0. ~ 1.
    image_tensor = image_tensor.unsqueeze(0).cuda()        # [1, 3, 512, 512]
    encoded = vae.get_codebook_indices(image_tensor)   # [1024]
    indices_list = encoded.squeeze().cpu().detach().tolist()
    dict_by_dicomid[dicom_id] = indices_list

with open('/home/edlab/jylee/Scaleup/data/mimiccxr_vqgan1024_res512_codebook_indices.pickle', 'wb') as f:
    pickle.dump(dict_by_dicomid, f)