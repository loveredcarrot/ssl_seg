# -*- coding: utf-8 -*- 
# @Time : 2021/2/4 15:08 
# @Author : aurorazeng
# @File : cv2_dataset.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.


# standard library
import os
import random
import math

# part packages
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import albumentations as albu
import matplotlib.pyplot as plt
import torchvision
import albumentations.pytorch.transforms


# assume you have txt file(image_path,label_path)
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, csv_path=None, split='val', crop_size=(512, 512), num=None, target_mpp=None):
        self.base_dir = base_dir
        self.csv_path = os.path.join(base_dir, csv_path)
        self.crop_size = crop_size
        self.sample_list = []
        self.split = split
        self.target_mpp = target_mpp
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == 'train':
            random.shuffle(self.sample_list)
            self.sample_list = self.sample_list[:max(len(self.sample_list), num)]
        print("total {} samples in {} set".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = case.split(',')[0]
        label_path = case.split(',')[1]
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.isfile(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        aug = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.1,
                                          contrast_limit=0.1, p=0.5),
            albu.Flip(always_apply=False, p=0.5),
            albu.HueSaturationValue(hue_shift_limit=5,
                                    sat_shift_limit=5,
                                    val_shift_limit=20, p=0.5),
            albu.Rotate(limit=30),
            albu.RandomCrop(self.crop_size[0], self.crop_size[1]),
            # albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # albumentations.pytorch.transforms.ToTensorV2(),
        ])
        if self.split == 'train':
            imagetrans = aug(image=image, mask=label)
            image = imagetrans['image']
            label = imagetrans['mask']
        else:
            mpp = os.path.basename(image_path).split('.')[-2]
            if '_' in mpp:
                mpp = 424.0
            else:
                mpp = float(mpp)
            scale = round(mpp / 1000 / self.target_mpp, 4)
            resize_shape = int(math.ceil(image.shape[0] * scale / 64) * 64)
            image = cv2.resize(image, (resize_shape, resize_shape))
            label = cv2.resize(label, (resize_shape, resize_shape))

        image = image / 255
        image = image.transpose((2, 0, 1))
        label = label / 255
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        sample["idx"] = idx

        return sample


if __name__ == '__main__':
    db_train = BaseDataSets(base_dir=r"D:\mypythonproject\ssl_experiment\data",
                            csv_path='train.txt',
                            split='train',
                            crop_size=(512, 512),
                            num=None,
                            target_mpp=0.848)
    db_val = BaseDataSets(base_dir=r"D:\mypythonproject\ssl_experiment\data",
                          csv_path='val.txt',
                          split='val',
                          crop_size=(512, 512),
                          num=None,
                          target_mpp=0.848)
    trainloader = DataLoader(db_train, batch_size=1)
    valloader = DataLoader(db_val, batch_size=1)
    for i_batch, sampled_batch in enumerate(trainloader):
        if i_batch == 29:
            break
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        if i_batch == 1:
            print(torch.max(volume_batch), torch.min(volume_batch), torch.max(label_batch), torch.min(label_batch))
            print(volume_batch.shape, label_batch.shape)
        print('train', np.count_nonzero(label_batch), np.count_nonzero(label_batch == 1))
        img = volume_batch.cpu().clone()
        img = img.squeeze(0).numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        cv2.imwrite('D:\mypythonproject\ssl_experiment\data\{}_train_image.jpg'.format(i_batch),
                    (img * 255)[:, :, ::-1])

        lab = label_batch.cpu().clone()
        lab = lab.numpy()
        lab = np.transpose(lab, (1, 2, 0))
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(lab, cmap='gray')
        plt.show()
        cv2.imwrite('D:\mypythonproject\ssl_experiment\data\{}_train_mask.jpg'.format(i_batch), lab * 255)
    for i_batch, sampled_batch in enumerate(valloader):
        if i_batch == 10:
            break
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        if i_batch == 1:
            print(torch.max(volume_batch), torch.min(volume_batch), torch.max(label_batch), torch.min(label_batch))
            print(volume_batch.shape, label_batch.shape)
        print('val', np.count_nonzero(label_batch), np.count_nonzero(label_batch == 1))
        img = volume_batch.cpu().clone()
        img = img.squeeze(0).numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        cv2.imwrite('D:\mypythonproject\ssl_experiment\data\{}_val_image.jpg'.format(i_batch), (img * 255)[:, :, ::-1])

        lab = label_batch.cpu().clone()
        lab = lab.numpy()
        lab = np.transpose(lab, (1, 2, 0))
        plt.imshow(lab, cmap='gray')
        plt.show()
        cv2.imwrite('D:\mypythonproject\ssl_experiment\data\{}_val_mask.jpg'.format(i_batch), lab * 255)