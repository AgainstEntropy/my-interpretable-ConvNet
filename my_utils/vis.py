# -*- coding: utf-8 -*-
# @Date    : 2022/1/23 14:05
# @Author  : WangYihao
# @File    : vis.py
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from . import data
from .utils import get_device

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def grid_vis(loader, row_num, model=None):
    imgs, labels = next(iter(loader))
    if model is not None:
        device = get_device(model)
        scores = model(imgs.to(device))
        preds = scores.argmax(axis=1)

    vis = imgs.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
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
    """

    Args:
        act ():
        label ():
        row_num ():

    Returns:

    """
    # act = act.permute(1, 2, 3, 0)  # (1, C, H, W) -> (C, H, W, 1)
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
        plt.title(f"GT:{label} ({i}/{subfig_num - 1})")
        plt.imshow(act[i], 'gray')
        plt.axis('off')
    plt.tight_layout()


def Vis_cam(loader, model, target_layers, img_num=8, mode="heatmap_only"):
    """

    Args:
        mode ():
        loader ():
        model ():
        target_layers ():
        img_num ():

    Returns:

    """
    device = get_device(model)
    imgs, labels = next(iter(loader))
    input_tensors = imgs[:img_num].to(device)
    labels = labels[:img_num]

    model.eval()
    scores = model(input_tensors)
    preds = scores.argmax(axis=1)

    vis = imgs.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    col_num = img_num
    row_num = 5

    fig = plt.figure(figsize=(2 * img_num, 10))
    for col in range(col_num):
        input_tensor = input_tensors[[col]]
        plt.subplot(row_num, col_num, col + 1)
        plt.imshow(vis[col], "gray")
        plt.title(f"GT:{labels[col]}, pred:{preds[col]}")
        plt.axis("off")
        for row in range(1, 5):
            targets = [ClassifierOutputTarget(row - 1)]
            with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
                img_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            if mode == "heatmap_only":
                img_cam = cv2.applyColorMap(np.uint8(255 * img_cam), colormap=cv2.COLORMAP_JET)
            elif mode == "heatmap_on_img":
                img_cam = show_cam_on_image(vis[col].numpy() / 255, img_cam, use_rgb=True)
            plt.subplot(row_num, col_num, row * col_num + col + 1)
            # By default, a linear scaling mapping the lowest value to 0 and the highest to 1 is used in plt.imshow()
            plt.imshow(img_cam, "gray")
            plt.axis("off")


def Vis_mean(dataset_dir='/home/wangyh/01-Projects/03-my/Datasets/polygons_unfilled_32_2',
             dataset_type="train"):
    dataset = data.MyDataset(os.path.join(dataset_dir, dataset_type),
                             transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=len(dataset))
    imgs, labels = next(iter(loader))
    gray_imgs = imgs.squeeze()
    num_per_class = len(dataset) // 4

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        img_mean = gray_imgs[i * num_per_class: (i + 1) * num_per_class].mean(dim=0)
        ax = axs[i].imshow(img_mean, cmap="gray")
        axs[i].set_title(f'{i}')
    cb = fig.colorbar(ax, ax=axs, orientation='horizontal', location='bottom')
    plt.show()


def Vis_pca(dim=2,
            dataset_dir='/home/wangyh/01-Projects/03-my/Datasets/polygons_unfilled_32_2',
            dataset_type="train"):
    dataset = data.MyDataset(os.path.join(dataset_dir, dataset_type),
                             transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=len(dataset))
    imgs, labels = next(iter(loader))
    input_imgs = imgs.reshape(len(dataset), -1)

    x = PCA(dim).fit_transform(input_imgs)

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d') if dim == 3 else fig.add_subplot(121)
    if dim == 2:
        ax2 = fig.add_subplot(122, projection='3d') if dim == 3 else fig.add_subplot(122)
    for i in range(4):
        ax1.scatter(*[x[labels == i, _] for _ in range(dim)], label=f'{i + 3}')
        if dim == 2:
            ax2.scatter(*[x[labels == 3-i, _] for _ in range(dim)], label=f'{6 - i}', c=color_list[3-i])
    ax1.legend(title='edges #', loc='best')
    if dim == 2:
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[::-1], labels[::-1], title='edges #', loc='best')
    fig.suptitle(f"PCA(n={dim})")
    plt.show()


def vis_4D(data, figsize_factor=2, cmap='viridis'):
    """
    Visualize a 4D tensor with shape (N, C, H, W) using N rows and C columns.
    """
    assert len(data.shape) == 4
    row_num, col_num = data.shape[:2]
    plt.figure(figsize=(col_num * figsize_factor, row_num * figsize_factor))
    for idx, filer in enumerate(data.reshape((-1, *data.shape[2:]))):
        plt.subplot(row_num, col_num, idx + 1)
        plt.axis('off')
        plt.imshow(filer, cmap=cmap)

    plt.show()
