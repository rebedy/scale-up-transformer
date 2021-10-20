from tqdm import tqdm
import numpy as np
import albumentations
from PIL import Image
import os
import time

root = '/home/data_storage/dylee_135/mimic-cxr-jpg/2.0.0/files'
newroot = '/home/data_storage/dylee_135/mimic-cxr-jpg/2.0.0/resized_512'

f = open('data/mimiccxrtrain.txt', 'r')
trainfiles = f.read().splitlines()    # f.readlines()하면 맨 뒤 \n도 붙어 있는데 이건 없네
f.close()

f = open('data/mimiccxrvalidation.txt', 'r')
valfiles = f.read().splitlines()    # f.readlines()하면 맨 뒤 \n도 붙어 있는데 이건 없네
f.close()

rescaler = albumentations.SmallestMaxSize(max_size = 512)    # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
cropper = albumentations.CenterCrop(height=512, width=512)
preprocessor = albumentations.Compose([rescaler, cropper])   # input으로 numpy array만 받는다

def preprocess_image(image_path):
    image = Image.open(image_path)   # PIL format
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)         # value 0 ~ 255  [H, W, C]
    image = preprocessor(image=image)["image"]  # albumentations의 output: {'image': numpy.ndarray(256,256,3)}  uint8
    return image   # ndarray (256, 256, 3)  uint8

for filepath in tqdm(trainfiles):
    s = time.monotonic()
    image = preprocess_image( os.path.join(root, filepath) )  # ndarray (256, 256, 3)  uint8
    newpath = os.path.join(newroot, filepath)
    os.makedirs( os.path.split(newpath)[0], exist_ok=True )   # os.path.split(path): 폴더부분과 파일부분을 서로 잘라준다. os.path.join의 반대
    Image.fromarray(image).save(newpath)
    
    print(time.monotonic()-s)
    exit()
print("trainset finish")

for filepath in tqdm(valfiles):
    image = preprocess_image( os.path.join(root, filepath) )  # ndarray (256, 256, 3)  uint8
    newpath = os.path.join(newroot, filepath)
    os.makedirs( os.path.split(newpath)[0], exist_ok=True )   # os.path.split(path): 폴더부분과 파일부분을 서로 잘라준다. os.path.join의 반대
    Image.fromarray(image).save(newpath)
print("valset finish")