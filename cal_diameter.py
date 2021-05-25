import os

import numpy as np
from PIL import Image


def make_mask(mask_txt_path):
    imgs = []
    f_mask = open(mask_txt_path, 'r')

    for mask in f_mask:
        mask = mask.rstrip()
        imgs.append(mask)

    f_mask.close()
    return imgs

def get_diameter(train_mask_txt_path):
    imgs = make_mask(train_mask_txt_path)
    diameter = 0
    for i, img in enumerate(imgs):
        img = Image.open(img)
        img = np.array(img)
        dim0 = np.max(np.where(img == 128)[0]) - np.min(np.where(img == 128)[0])
        dim1 = np.max(np.where(img == 128)[1]) - np.min(np.where(img == 128)[1])
        dim_max = dim0 if dim0 > dim1 else dim1
        if dim_max > diameter:
            diameter =  dim_max
        print(i)
    return diameter

if __name__ == '__main__':
    train_mask_txt_path = os.path.join("refuge", "train_mask.txt")
    diameter = get_diameter(train_mask_txt_path)
    print(diameter)
