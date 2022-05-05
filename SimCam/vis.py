# -*- coding: utf-8 -*-
# @Date    : 2022/4/19 10:54
# @Author  : WangYihao
# @File    : vis.py

from matplotlib import pyplot as plt


def vis_cam(imgs, lebels, preds, cam_maps):
    imgs = imgs.cpu().squeeze()
    N = imgs.shape[0]
    fig, axs = plt.subplots(2, N, constrained_layout=True)
    for col in range(N):
        axs[0, col].imshow(imgs[col], cmap='gray')
        # axs[0, col].set_title()
        sim_map = axs[1, col].imshow(cam_maps[col], cmap='jet')
        # frame = plt.gca()
        # frame.axes.get_xaxis().set_visible(False)
        # frame.axes.get_yaxis().set_visible(False)

    fig.colorbar(sim_map, ax=axs[1, :], shrink=0.8, location='bottom')
    plt.show()
