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

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            # 获取类别文件夹下所有图片的路径
            img_list = os.listdir(i_dir)
            for i in range(len(img_list)):
                if img_list[i].startswith('.'):
                    continue
                # label = img_list[i][0]
                img_path = os.path.join("../", i_dir, img_list[i])
                line = img_path + '\n'
                f.write(line)
    f.close()

def make_dataset(img_txt_path, mask_txt_path):
    imgs = []
    f_img = open(img_txt_path, 'r')
    f_mask = open(mask_txt_path, 'r')

    for img in f_img:
        img = img.rstrip()
        # img = img.split()
        imgs.append([img])

    i = 0
    for mask in f_mask:
        mask = mask.rstrip()
        # mask_word = mask.split()
        imgs[i].append(mask)
        i += 1

    f_img.close()
    f_mask.close()
    return imgs

class REFUGESegmentation(data.Dataset):
    cmap = refuge_cmap()
    def __init__(self, img_txt_path, mask_txt_path, transform=None):
        imgs = make_dataset(img_txt_path, mask_txt_path)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)
        img_y = np.array(img_y)
        img_y[img_y < 255] = 1
        img_y[img_y == 255] = 0

        if self.transform is not None:
            img_x = np.array(img_x)
            aug = self.transform(image=img_x, mask=img_y)
            img_x = aug['image']
            img_y = aug['mask']

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


if __name__ == '__main__':
    train_txt_path = os.path.join("refuge", "train.txt")
    train_mask_txt_path = os.path.join("refuge", "train_mask.txt")
    
    train_dir = os.path.join("refuge", "Training400")
    train_mask_dir = os.path.join(
        "refuge", "Training400_Mask", "Disc_Cup_Masks")

    valid_txt_path = os.path.join("refuge", "valid.txt")
    valid_mask_txt_path = os.path.join("refuge", "valid_mask.txt")

    valid_dir = os.path.join("refuge", "Validation400")
    valid_mask_dir = os.path.join("refuge", "Validation400_Mask")

    # test_txt_path = os.path.join("refuge", "test.txt")
    # test_mask_txt_path = os.path.join("refuge", "test_mask.txt")

    gen_txt(train_txt_path, train_dir)
    gen_txt(train_mask_txt_path, train_mask_dir)
    
    gen_txt(valid_txt_path, valid_dir)
    gen_txt(valid_mask_txt_path, valid_mask_dir)
