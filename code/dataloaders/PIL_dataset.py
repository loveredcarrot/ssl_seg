# -*- coding: utf-8 -*- 
# @Time : 2021/2/4 15:07 
# @Author : aurorazeng
# @File : PIL_dataset.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

# standard library
import os
import glob
import random
from PIL import Image
import math

# part packages
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data.sampler import Sampler
import itertools
import time


# assume you have txt file(image_path,label_path)
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, csv_path=None, split='val', crop_size=(512, 512), num=None, target_mpp=None):
        self.base_dir = base_dir
        self.csv_path = os.path.join(base_dir, csv_path)
        self.split = split
        self.crop_size = crop_size
        self.target_mpp = target_mpp
        self.sample_list = []
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == 'train':
            self.sample_list = self.sample_list[:min(len(self.sample_list), num)]
        print("total {} samples in {} set".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = case.split(',')[0]
        label_path = case.split(',')[1]
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert('RGB')
        if os.path.isfile(label_path):
            label = Image.open(label_path).convert('L')
        if self.split == "train":
            # 颜色变换
            process = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = process(image)

            image, label = random_flip(image, label)
            image, label = random_rotate(image, label)
            image, label = random_crop(image, label, self.crop_size)
        else:
            mpp = os.path.basename(image_path).split('.')[-2]
            if '_' in mpp:
                mpp = 424.0
            else:
                mpp = float(mpp)
            scale = round(mpp / 1000 / self.target_mpp, 4)
            resize_shape = int(math.ceil(image.size[0] * scale / 64) * 64)
            image = image.resize((resize_shape, resize_shape))
            label = label.resize((resize_shape, resize_shape))
        image = tf.to_tensor(image).type(torch.FloatTensor)
        label = tf.to_tensor(label).type(torch.ByteTensor)
        label = label.squeeze(0)
        sample = {'image': image, 'label': label}
        sample["idx"] = idx
        return sample


# data_augment
def random_rotate(image, label):
    angle = transforms.RandomRotation.get_params([-180, 180])
    image = image.rotate(angle)
    label = label.rotate(angle)
    return image, label


def random_flip(image, label):
    if random.random() > 0.5:
        image = tf.hflip(image)
        label = tf.hflip(label)
    if random.random() > 0.5:
        image = tf.vflip(image)
        label = tf.vflip(label)
    return image, label


def random_crop(image, label, crop_size):
    i, j, h, w = transforms.RandomCrop.get_params(image, (crop_size[0], crop_size[1]))
    image = tf.crop(image, i, j, h, w)
    label = tf.crop(label, i, j, h, w)
    return image, label


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

# https://arxiv.org/abs/1906.01916v5
class RepeatSampler(Sampler):
    r"""Repeated sampler

    Arguments:
        data_source (Dataset): dataset to sample from
        sampler (Sampler): sampler to draw from repeatedly
        repeats (int): number of repetitions or -1 for infinite
    """

    def __init__(self, sampler, repeats=-1):
        if repeats < 1 and repeats != -1:
            raise ValueError('repeats should be positive or -1')
        self.sampler = sampler
        self.repeats = repeats

    def __iter__(self):
        if self.repeats == -1:
            reps = itertools.repeat(self.sampler)
            return itertools.chain.from_iterable(reps)
        else:
            reps = itertools.repeat(self.sampler, self.repeats)
            return itertools.chain.from_iterable(reps)

    def __len__(self):
        if self.repeats == -1:
            return 2 ** 62
        else:
            return len(self.sampler) * self.repeats


if __name__ == '__main__':
    db_train = BaseDataSets(base_dir=r"D:\mypythonproject\ssl_experiment\data",
                            csv_path='train.txt',
                            split='train',
                            crop_size=(512, 512),
                            num=10,
                            target_mpp=0.848)
    db_val = BaseDataSets(base_dir=r"D:\mypythonproject\ssl_experiment\data",
                          csv_path='val.txt',
                          split='val',
                          crop_size=(512, 512),
                          num=None,
                          target_mpp=0.848)
    trainloader = DataLoader(db_train, batch_size=1, num_workers=3)
    valloader = DataLoader(db_val, batch_size=1)
    for i_batch, sampled_batch in enumerate(trainloader):
        if i_batch == 29:
            break
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        if i_batch == 0:
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
        if i_batch == 0:
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
