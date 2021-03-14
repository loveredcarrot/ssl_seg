# -*- coding: utf-8 -*- 
# @Time : 2021/2/5 15:18 
# @Author : aurorazeng
# @File : val_2D.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.


import numpy as np
import torch
from medpy import metric

"""
Calculate  Two categories some metric
sensitive = TP/ (TP + FP)  TP/P
specificity = TN /(TN + FN) = TN /N
dice(F-score)
F1 = 2TP/(2TP+FP+FN)
"""


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    else:
        return 0, 500, 500


# def test_single_volume(image, label, net):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     input = torch.from_numpy(image).unsqueeze(
#         0).float().cuda()
#     net.eval()
#     with torch.no_grad():
#         out = torch.argmax(torch.softmax(
#             net(input), dim=1), dim=1).squeeze(0)
#         out = out.cpu().detach().numpy()
#         prediction = out
#     total_pred = np.count_nonzero(prediction)
#     total_true = np.count_nonzero(label)
#     tp = np.count_nonzero(prediction + label == 2)
#     fp = total_pred - tp
#     fn = total_true - tp
#     if tp < 0 or fp < 0 or fn < 0:
#         return False
#     f1 = tp / (tp + 0.5 * (fp + fn) + 1e-5)
#     return f1

def test_single_volume(image, label, net):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = out
    metric_list = calculate_metric_percase(prediction, label)
    return metric_list
