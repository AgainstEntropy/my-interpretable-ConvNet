# -*- coding: utf-8 -*-
# @Date    : 2022/1/23 14:05
# @Author  : WangYihao
# @File    : vis.py

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import get_device


def grid_vis(loader, row_num, model=None):
    imgs, labels = next(iter(loader))
    if model is not None:
        device = get_device(model)
        scores = model(imgs.to(device))
        preds = scores.argmax(axis=1)

    vis = imgs.permute(0, 2, 3, 1)
    batch_size = vis.size(0)
    if row_num ** 2 < batch_size:
        subfig_num = row_num ** 2
        col_num = row_num
    else:
        subfig_num = batch_size
        row_num = int(np.sqrt(batch_size))
        col_num = batch_size // row_num + 1
    fig = plt.figure()
    for i in range(subfig_num):
        plt.subplot(row_num, col_num, i + 1)
        if model is not None:
            plt.title(f"{labels[i]} : {preds[i]}")
        else:
            plt.title(f"GT:{labels[i]}")
        plt.imshow(vis[i], 'gray')
        plt.axis('off')
    plt.tight_layout()
    # plt.show()


def vis_act(act, label, row_num=6):
    act = act.permute(1, 2, 3, 0)  # (1, C, H, W) -> (C, H, W, 1)
    chans_num = act.size(0)
    if row_num ** 2 < chans_num:
        subfig_num = row_num ** 2
        col_num = row_num
    else:
        subfig_num = chans_num
        row_num = int(np.sqrt(chans_num))
        col_num = chans_num // row_num + 1
    fig = plt.figure()
    for i in range(subfig_num):
        plt.subplot(row_num, col_num, i + 1)
        plt.title(f"GT:{label}")
        plt.imshow(act[i], 'gray')
        plt.axis('off')
    plt.tight_layout()
