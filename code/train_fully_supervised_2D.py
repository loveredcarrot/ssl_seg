# -*- coding: utf-8 -*- 
# @Time : 2021/2/5 16:22 
# @Author : aurorazeng
# @File : train_mean_teacher_2D.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.


import argparse
import logging
import os
import random
import shutil
import sys
import time

sys.path.append("..")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import itertools

from dataloaders.PIL_dataset import (BaseDataSets, TwoStreamBatchSampler, RepeatSampler)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str,
                    default='../csv_data/', help='root path of csv_data')
parser.add_argument('--exp', type=str,
                    default='/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='LinkNetBaseWithDrop', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--num_epochs', type=int,
                    default=300, help='max epoch number to train')
parser.add_argument('--iters_per_epoch', type=int,
                    default=100, help='max epoch number to train')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[512, 512],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--normalize', type=bool, default=False,
                    help='whether use normalize')

# train data path and val data path
parser.add_argument('--labeled_num', type=int, default=86,
                    help='labeled_data')
parser.add_argument('--labeled_csv', type=str, default='train_label_86.txt',
                    help='labeled_csv path')
parser.add_argument('--val_data', type=str, default='val_no_negative.txt',
                    help='val_data_path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(args, snapshot_path):
    # Load parameter
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.num_epochs * args.iters_per_epoch
    max_epochs = args.num_epochs
    labeled_case = args.labeled_num
    iters_per_epoch = args.iters_per_epoch
    is_normalize = args.normalize

    # Build network
    model = net_factory(net_type=args.model, in_chns=3,
                        class_num=num_classes)
    model = model.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Load data sets
    db_sup_train = BaseDataSets(base_dir=args.csv_path,
                                csv_path=args.labeled_csv, split='train',
                                crop_size=args.patch_size,
                                num=None, target_mpp=0.848,
                                is_normalize=is_normalize)

    db_val = BaseDataSets(base_dir=args.csv_path,
                          csv_path=args.val_data, split='val',
                          num=None, target_mpp=0.848,
                          is_normalize=is_normalize)

    print("Total slices is: {}, labeled slices is: {}".format(
        8548, args.labeled_csv))

    # train data and val data pipeline: data loaders
    train_sup_loader = DataLoader(db_sup_train, batch_size=batch_size,
                                  shuffle=True, num_workers=16,
                                  pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=6)

    # switch to train mode
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(iters_per_epoch))

    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(max_epochs), ncols=70)
    for epoch_num in iterator:

        for i_batch, labeled_sampled_batch in enumerate(train_sup_loader):
            # measure data loading time and control load data time
            if i_batch >= 100:
                break

            labeled_volume_batch, label_batch = labeled_sampled_batch['image'], labeled_sampled_batch['label']
            labeled_volume_batch, label_batch = labeled_volume_batch.cuda(), label_batch.cuda()

            labeled_outputs = model(labeled_volume_batch)
            labeled_outputs_soft = torch.softmax(labeled_outputs, dim=1)

            # supervised_loss
            loss_ce = ce_loss(labeled_outputs, label_batch[:].long())
            loss_dice = dice_loss(labeled_outputs_soft, label_batch.unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            # total loss
            loss = supervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update  learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num > 0 and iter_num % 500 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                writer.add_scalar('info/val_dice', metric_list[0], iter_num)
                writer.add_scalar('info/val_hd95', metric_list[1], iter_num)
                writer.add_scalar('info/val_asd', metric_list[2], iter_num)

                performance = metric_list[0]
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    time.sleep(1)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f ' % (iter_num, performance))
                logging.info(
                    'iteration %d : mean_hd95 : %f ' % (iter_num, metric_list[1]))
                logging.info(
                    'iteration %d : mean_asd : %f ' % (iter_num, metric_list[2]))
                model.train()

            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                time.sleep(1)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
        time.sleep(0.003)
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
