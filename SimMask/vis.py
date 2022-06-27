# -*- coding: utf-8 -*-
# @Date    : 2022/4/19 10:54
# @Author  : WangYihao
# @File    : vis.py
import numpy as np
from matplotlib import pyplot as plt, colors

from my_utils.utils import fig2array


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


def vis_4D_plot(data, title: str = None, norm=True, figsize_factor=1, tune_factor=0, fontsize=16, cmap='viridis',
                return_mode=None):
    """
    Visualize a 4D tensor with shape (N, C, H, W) using N rows and C columns.
    """
    assert len(data.shape) == 4
    row_num, col_num = data.shape[:2]
    fig, axes = plt.subplots(row_num, col_num,
                             figsize=(col_num * figsize_factor + tune_factor, row_num * figsize_factor),
                             constrained_layout=True)
    fig.patch.set_facecolor('none')
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=0.1, wspace=0.1)
    if row_num == 1:
        axes = np.array([axes])
    if col_num == 1:
        axes = np.array([np.array([ax]) for ax in axes])
    images = []
    for row in range(row_num):
        for col in range(col_num):
            images.append(axes[row, col].imshow(data[row, col], cmap=cmap))
            axes[row, col].set_axis_off()

    if norm:
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        # fig.colorbar(images[0], ax=axes, orientation='vertical')

    if return_mode is None:
        plt.show()
    elif return_mode == 'plt_fig':
        return fig
    elif return_mode == 'fig_array':
        return fig2array(fig)
