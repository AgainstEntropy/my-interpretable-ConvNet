# -*- coding: utf-8 -*-
# @Date    : 2022/4/19 09:41
# @Author  : WangYihao
# @File    : utils.py

import torch


def adaptive_softmax(scores, normalize):
    negative_mask = scores < 0
    scores = scores.__abs__().float()
    if normalize:
        scores /= scores.max()
    weights = torch.softmax(scores, dim=0)
    weights[negative_mask] *= -1

    return weights


def zoom_to_01(x):
    return (x - x.min()) / (x.max() - x.min())
