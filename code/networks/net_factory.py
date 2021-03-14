# -*- coding: utf-8 -*- 
# @Time : 2021/2/5 15:44 
# @Author : aurorazeng
# @File : net_factory.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

from networks.linknet import LinkNet, LinkNetBase, LinkNetBaseWithDrop

def net_factory(net_type="unet", in_chns=3, class_num=2):
    # if net_type == "unet":
    #     net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "LinkNetBase":
        net = LinkNetBase(n_classes=class_num).cuda()
    elif net_type == "LinkNet":
        net = LinkNet(n_classes=class_num).cuda()
    elif net_type == "LinkNetBaseWithDrop":
        net = LinkNetBaseWithDrop(n_classes=class_num).cuda()
    else:
        net = None
    return net
