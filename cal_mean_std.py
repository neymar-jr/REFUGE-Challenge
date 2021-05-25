import os
import numpy as np
from PIL import Image
import glob

# 得到包含训练集所有图片4D张量
def train4D(data_path):
    tensor = []
    images = glob.glob(os.path.join(data_path, '*[0-9].png'))
    for i, image in enumerate(images):
        print(i)
        pic = Image.open(image).convert('RGB')
        pic = np.array(pic).transpose(2, 0, 1)
        tensor.append(pic)
    return np.array(tensor)

# 计算训练集的均值和方差用于标准化
def calculate(tensor):
    mean = []
    std = []
    for i in range(3):
        mean.append(np.mean(tensor[:, i]))
        std.append(np.std(tensor[:, i]))
    return np.array(mean), np.array(std)

if __name__ == '__main__':

    train_data_path = os.path.join("../", "refuge", "Train_edges")
    tensor = train4D(train_data_path)
    mean, std = calculate(tensor)
    print('mean:{0}, std:{1}'.format(mean / 255., std / 255.))