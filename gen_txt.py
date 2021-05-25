import os

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
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + '\n'
                f.write(line)
    f.close()


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

    # 为数据集生成对应txt文件以及划分训练集，验证集，测试集
    gen_txt(train_txt_path, train_dir)
    gen_txt(train_mask_txt_path, train_mask_dir)

    gen_txt(valid_txt_path, valid_dir)
    gen_txt(valid_mask_txt_path, valid_mask_dir)
