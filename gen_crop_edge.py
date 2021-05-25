import os

import numpy as np
from PIL import Image
from PIL import ImageFilter


def gen_edge(img_txt_path, mask_txt_path):
    imgs = []
    img_names = []

    f_img = open(img_txt_path, 'r')
    f_mask = open(mask_txt_path, 'r')

    for img in f_img:
        img = img.rstrip()
        imgs.append([img])
        img_names.append([img.split('/')[-1]])

    i = 0
    for mask in f_mask:
        mask = mask.rstrip()
        imgs[i].append(mask)
        img_names[i].append(mask.split('/')[-1])
        i += 1

    f_img.close()
    f_mask.close()

    for i in range(len(imgs)):

        x_path, y_path = imgs[i]
        x_name, y_name = img_names[i]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)

        img_y = np.array(img_y)

        img_y[img_y == 0] = 2
        img_y[img_y == 128] = 1
        img_y[img_y == 255] = 0

        loc = np.where(img_y == 1)
        center_dim0 = (np.max(loc[0]) + np.min(loc[0])) // 2
        center_dim1 = (np.max(loc[1]) + np.min(loc[1])) // 2

        img_y = Image.fromarray(img_y)

        img_mask = img_y

        # crop size (512, 512)
        if center_dim1 >= 256 and center_dim0 >= 256:
            img_x = img_x.crop((center_dim1 - 256, center_dim0 - 256, center_dim1 + 256, center_dim0 + 256))
            img_y = img_y.crop((center_dim1 - 256, center_dim0 - 256, center_dim1 + 256, center_dim0 + 256))
        else:
            img_x_crop = img_x.crop((0, center_dim0 - 256, center_dim1 + 256, center_dim0 + 256))
            img_y_crop = img_y.crop((0, center_dim0 - 256, center_dim1 + 256, center_dim0 + 256))

            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 0)

            img_black_rgb.paste(img_x_crop, (256 - center_dim1, 0, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (256 - center_dim1, 0, 512, 512))
            img_y = img_white_gray

        edge = np.array(img_y.filter(ImageFilter.Kernel((5, 5), (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24,
                                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1), 1, 0)))
        img_edge = np.zeros((512, 512))
        img_edge[edge > 0] = 255
        img_edge = Image.fromarray(img_edge)
        
        train_path = os.path.join("refuge", "Train_crop_edge")
        valid_path = os.path.join("refuge", "Valid_crop_edge")

        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(valid_path):
            os.mkdir(valid_path)

        path = train_path
        # path = valid_path
        
        img_x.convert('RGB').save(os.path.join(path + x_name.split('.')[0] + '.png'))
        img_edge.convert('L').save(os.path.join(path + y_name.split('.')[0] + '_edge.png'))
        img_mask.convert('L').save(os.path.join(path + y_name.split('.')[0] + '_mask.png'))

        print(i)
    




if __name__ == '__main__':

    train_txt_path = os.path.join("../", "refuge", "train.txt")
    train_mask_txt_path = os.path.join("../", "refuge", "train_mask.txt")

    valid_txt_path = os.path.join("../", "refuge", "valid.txt")
    valid_mask_txt_path = os.path.join("../", "refuge", "valid_mask.txt")

    gen_edge(train_txt_path, train_mask_txt_path)
    # gen_edge(valid_txt_path, valid_mask_txt_path)

