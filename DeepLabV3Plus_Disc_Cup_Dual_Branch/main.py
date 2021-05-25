import os
import random

import albumentations as A
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils import data
from tqdm import tqdm

import losses_pytorch
import network
import utils
from datasets import REFUGESegmentation
from metrics import StreamSegMetrics1, StreamSegMetrics2
from utils.config import get_argparser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'refuge':
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5), # 水平翻转
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.Normalize(mean=[0.508, 0.294, 0.147], std=[0.230, 0.153, 0.090]),
            ToTensorV2(), 
        ])
        
        val_transform = A.Compose([
            A.Normalize(mean=[0.508, 0.294, 0.147], std=[0.230, 0.153, 0.090]),
            ToTensorV2(), 
        ])

        train_data_path = os.path.join("../", "refuge", "Train_crop_edge")
        valid_data_path = os.path.join("../", "refuge", "Valid_crop_edge")

        train_dst = REFUGESegmentation(
            train_data_path, transform=train_transform)
        val_dst = REFUGESegmentation(
            valid_data_path, transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metric1, metric2):
    """Do validation and return specified samples"""
    metric1.reset()
    metric2.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.508, 0.294, 0.147],
                                   std=[0.230, 0.153, 0.090])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels, edges) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            edges = edges.to(device, dtype=torch.long)

            outputs = model(images)
            pred1 = outputs[:, :2].detach().max(dim=1)[1].cpu().numpy()
            pred2 = outputs[:, 2:].detach().max(dim=1)[1].cpu().numpy()
            target1 = edges.cpu().numpy()
            target2 = labels.cpu().numpy()

            metric1.update(target1, pred1)
            metric2.update(target2, pred2)

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    tar1 = target1[i]
                    tar2 = target2[i]
                    pre1 = pred1[i]
                    pre2 = pred2[i]

                    image = (denorm(image) * 255).transpose(1,
                                                            2, 0).astype(np.uint8)
                    tar1 = loader.dataset.decode_target(
                        tar1).astype(np.uint8)
                    tar2 = loader.dataset.decode_target(
                        tar2).astype(np.uint8)
                    pre1 = loader.dataset.decode_target(pre1).astype(np.uint8)
                    pre2 = loader.dataset.decode_target(pre2).astype(np.uint8)

                    Image.fromarray(image).save(
                        'results/%d_image.png' % img_id)
                    Image.fromarray(tar1).save(
                        'results/%d_edge.png' % img_id)
                    Image.fromarray(tar2).save(
                        'results/%d_mask.png' % img_id)
                    Image.fromarray(pre1).save(
                        'results/%d_edge_pred.png' % img_id)
                    Image.fromarray(pre2).save(
                        'results/%d_mask_pred.png' % img_id)

                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pre1, alpha=0.5)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_edge_overlay.png' %
                                img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pre2, alpha=0.5)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_mask_overlay.png' %
                                img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    img_id += 1

        score1 = metric1.get_results()
        score2 = metric2.get_results()
    return score1, score2

def validate2(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.508, 0.294, 0.147],
                                   std=[0.230, 0.153, 0.090])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels, edges) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1,
                                                            2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(
                        target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(
                        'results/%d_image.png' % img_id)
                    Image.fromarray(target).save(
                        'results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.5)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' %
                                img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score

def main():
    opts = get_argparser().parse_args()

    if opts.dataset.lower() == 'refuge':
        opts.num_classes = 3

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'fusion_rcf_resnet50': network.fusion_rcf_resnet50,
        'fusion_hed_resnet50': network.fusion_hed_resnet50,
        'fusion_case_resnet50': network.fusion_case_resnet50,
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet50_cbam': network.deeplabv3plus_resnet50_cbam,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_resnet101_cbam': network.deeplabv3plus_resnet101_cbam,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride)

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
        network.convert_to_separable_conv(seg.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metric1 = StreamSegMetrics1(2)
    metric2 = StreamSegMetrics2(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.Adam(params=model.parameters())
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.step_size, gamma=0.1)

    weight = torch.from_numpy(np.array([0.02, 1.0])).float().to(device)
    criterion1 = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion2 = losses_pytorch.FocalLoss()
    elif opts.loss_type == 'cross_entropy':
        criterion2 = nn.CrossEntropyLoss(reduction='mean')
        # criterion2 = losses_pytorch.TopKLoss()
    elif opts.loss_type == 'dice_loss':
        criterion2 = utils.BatchSoftDiceLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    if opts.test_only:
        seg = network.deeplabv3plus_resnet50(num_classes=opts.num_classes, output_stride=opts.output_stride)
        
        dict_trained = checkpoint["model_state"]
        dict_new = seg.state_dict().copy()
        list_trained = list(dict_trained.keys())
        list_new = list(seg.state_dict().keys())
        num = len(list_trained) - 10

        for i in range(num):
            dict_new[ list_new[i] ] = dict_trained[ list_trained[i] ]

        seg.load_state_dict(dict_new)
        seg = nn.DataParallel(seg)
        seg.to(device)
        seg.eval()

        val_score = validate2(
            opts=opts, model=seg, loader=val_loader, device=device, metrics=metric2)
        print(metric2.to_str(val_score))
        return

    alpha = 0.2
    beta = 1.0

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, edges) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            edges = edges.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = alpha * \
                criterion1(outputs[:, :2], edges) + beta * \
                criterion2(outputs[:, 2:], labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score1, val_score2 = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metric1=metric1, metric2=metric2)
                print(metric1.to_str(val_score1))
                print(metric2.to_str(val_score2))

                # 暂时只根据原图miou判断最佳模型

                if val_score2['Mean IoU'] > best_score:  # save best model
                    best_score = val_score2['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d_%.1f_%.1f_%.2f_%.1f_%s.pth' %
                              (opts.model, opts.dataset, opts.output_stride, alpha, beta, weight[0], weight[1], opts.attention))
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
