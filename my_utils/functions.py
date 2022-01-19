# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:39
# @Author  : WangYihao
# @File    : functions.py


import torch
from torch import nn
import torch.nn.functional as F


def Conv_BN_Relu(in_channel, out_channel, kernel_size=(3, 3), stride=None):
    if stride is not None:
        conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=False)
    else:
        conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding='same', bias=False)
    return nn.Sequential(
        conv_layer,
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )
