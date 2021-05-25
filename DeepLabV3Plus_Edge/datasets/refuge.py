import glob
import os

import numpy as np
import torch.utils.data as data
from PIL import Image


def refuge_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class REFUGESegmentation(data.Dataset):
    cmap = refuge_cmap()
    def __init__(self, data_path, transform=None):
        images = glob.glob(os.path.join(data_path, '*[0-9].png'))
        labels = glob.glob(os.path.join(data_path, '*_mask.png'))
        images.sort()
        labels.sort()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        path_x = self.images[index]
        path_y = self.labels[index]
        
        img_x = Image.open(path_x).convert('RGB')
        img_y = Image.open(path_y)

        img_x = np.array(img_x)
        img_y = np.array(img_y)

        img_y[img_y == 255] = 1

        if self.transform is not None:
            aug = self.transform(image=img_x, mask=img_y)
            img_x = aug['image']
            img_y = aug['mask']
        
        return img_x, img_y

    def __len__(self):
        return len(self.images)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


if __name__ == '__main__':
    pass
