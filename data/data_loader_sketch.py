from io import BytesIO

import numpy as np
import lmdb
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset
from data.tps_transformation import tps_transform

def RGB2Lab(inputs):
    return color.rgb2lab(inputs)

def Normalize(inputs):
    # output l [-50,50] ab[-128,128]
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    # ab = ab
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')

def selfnormalize(inputs):
    d = torch.max(inputs) - torch.min(inputs)
    out = (inputs) / d
    return out

def to_gray(inputs):
    img_gray = np.clip((np.concatenate((inputs[:,:,:1], inputs[:,:,:1], inputs[:,:,:1]), 2)+50)/100*255, 0, 255).astype('uint8')
    
    return img_gray

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img_src = np.array(img) # [0,255] uint8

        # ima_a = img_src
        # ima_a = ima_a.astype('uint8')
        # ima_a = Image.fromarray(ima_a)
        # ima_a.show()

        # get the left color image
        img_ref = img_src[:, :256]
        ## add gaussian noise
        noise = np.random.uniform(-5, 5, np.shape(img_ref))
        img_ref = np.clip(np.array(img_ref) + noise, 0, 255)


        img_ref = tps_transform(img_ref) # [0,255] uint8
        img_ref = np.clip(img_ref, 0, 255)
        img_ref = img_ref.astype('uint8')
        img_ref = Image.fromarray(img_ref)
        img_ref = np.array(self.transform(img_ref)) # [0,255] uint8

        img_lab = img_src[:, :256]
        img_lab = Normalize(RGB2Lab(img_lab)) # l [-50,50] ab [-128, 128]

        img_lab_sketch = img_src[:, 256:]
        img_lab_sketch = Normalize(RGB2Lab(img_lab_sketch)) # l [-50,50] ab [-128, 128]

        img = img_src[:, :256].astype('float32') # [0,255] float32 RGB
        img_ref = img_ref.astype('float32') # [0,255] float32 RGB

        # ima_a = img
        # ima_a = ima_a.astype('uint8')
        # ima_a = Image.fromarray(ima_a)
        # ima_a.show()
        #
        # ima_a = img_ref
        # ima_a = ima_a.astype('uint8')
        # ima_a = Image.fromarray(ima_a)
        # ima_a.show()
        #
        # ima_a = img_lab
        # ima_a = ima_a.astype('uint8')
        # ima_a = Image.fromarray(ima_a)
        # ima_a.show()


        img = numpy2tensor(img)
        img_ref = numpy2tensor(img_ref) # [B, 3, 256, 256]
        img_lab = numpy2tensor(img_lab)
        img_lab_sketch = numpy2tensor(img_lab_sketch)

        return img, img_ref, img_lab, img_lab_sketch
        
    