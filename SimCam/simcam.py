# -*- coding: utf-8 -*-
# @Date    : 2022/4/19 08:58
# @Author  : WangYihao
# @File    : simcam.py

from torchvision.transforms.functional import resize
from matplotlib import pyplot as plt

from .functional import adaptive_softmax, zoom_to_01


def SimCam(vis_model, imgs, layer=-1, softmax=True, normalize=True):
    scores, mid_outputs = vis_model(imgs)

    map_size = mid_outputs[layer].shape[-2:]

    null_imgs = resize(imgs, size=map_size).cpu()  # (N, C, H, W) -> (N, C, 20, 20)
    sims = (mid_outputs[layer] * null_imgs).sum(dim=(-2, -1))  # (N, C, 20, 20) -> (N, C)
    if softmax:
        sims = adaptive_softmax(sims, normalize)

    sim_maps = (sims[:, :, None, None] * mid_outputs[layer]).sum(dim=1)  # (N, C, 20, 20) -> (N, 20, 20)
    for i, sim_map in enumerate(sim_maps):
        sim_maps[i] = zoom_to_01(sim_map)

    return sim_maps
