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
        edges = glob.glob(os.path.join(data_path, '*_edge.png'))
        images.sort()
        labels.sort()
        edges.sort()
        self.images = images
        self.labels = labels
        self.edges = edges
        self.transform = transform

    def __getitem__(self, index):
        path_x = self.images[index]
        path_y = self.labels[index]
        path_z = self.edges[index]
        
        img_x = Image.open(path_x).convert('RGB')
        img_y = Image.open(path_y)
        img_z = Image.open(path_z)

        img_x = np.array(img_x)
        img_y = np.array(img_y)
        img_z = np.array(img_z)
        
        img_y[img_y == 0] = 2
        img_y[img_y == 128] = 1
        img_y[img_y == 255] = 0

        img_z[img_z == 255] = 1

        if self.transform is not None:
            aug1 = self.transform(image=img_x, mask=img_y)
            aug2 = self.transform(image=img_x, mask=img_z)
            img_x = aug1['image']
            img_y = aug1['mask']
            img_z = aug2['mask']
        
        return img_x, img_y, img_z

    def __len__(self):
        return len(self.images)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


if __name__ == '__main__':
    pass
