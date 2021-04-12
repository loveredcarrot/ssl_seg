# -*- coding: utf-8 -*- 
# @Time : 2021/3/4 14:55 
# @Author : aurorazeng
# @File : test_2D.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

import argparse
import os
import shutil
import math

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric

# from networks import linkNetBase
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../csv_data', help='path of data')
parser.add_argument('--exp', type=str,
                    default='Fully_Supervised_2_23_N_100_labeled', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='LinkNetBase', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=1522,
                    help='labeled data')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    return dice


def test_single_volume(case_path, net, test_save_path, FLAGS):
    image_path = case_path.split(',')[0]
    label_path = case_path.split(',')[1]
    image = Image.open(image_path).convert('RGB')
    label = Image.open(label_path).convert('L')
    mpp = os.path.basename(image_path).split('.')[-2]
    if '_' in mpp:
        mpp = 424.0
    else:
        mpp = float(mpp)
    scale = round(mpp / 1000 / 0.848, 4)
    resize_shape = int(math.ceil(image.size[0] * scale / 64) * 64)
    image = image.resize((resize_shape, resize_shape))
    label = label.resize((resize_shape, resize_shape))
    image = np.asarray(image, np.float32)
    label = np.asarray(label, np.float32)
    image /= 255
    label /= 255
    label = label.astype(np.int)
    image = np.transpose(image, (2, 0, 1))
    prediction = np.zeros_like(label)
    input = torch.from_numpy(image).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = out
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/val.txt', 'r') as f:
        case_list = f.readlines()
    case_list = [item.replace('\n', '')
                 for item in case_list]
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "../model/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=3,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    first_metric_list = []
    target_txt = 'dice.txt'
    f = open(test_save_path + target_txt, 'w')
    for case in tqdm(case_list):
        first_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_metric_list.append(first_metric)
        f.write(case + ',_' + str(first_metric) + '\n')
        first_total += first_metric

    avg_metric = first_total / len(case_list)
    return avg_metric
    f.close()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
