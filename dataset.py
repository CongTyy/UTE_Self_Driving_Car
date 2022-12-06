import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import random




def augment_hsv(im, hgain= 0, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


# Change cv2.resize to import data correctly

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, width, height):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.width, self.height = width, height

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index])
        image = cv2.resize(image, (self.width, self.height), interpolation = cv2.INTER_LINEAR)
        augment_hsv(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.0 ## (512, 512, 3)
        
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        
        image = image.astype(np.float32)
        
        image = torch.from_numpy(image)
        
        
        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask
    def __len__(self):
        return self.n_samples
    

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

