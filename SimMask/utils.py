import os
import time
from typing import Union, Optional

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt, colors
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from tqdm import tqdm


def get_device(model):
    return next(model.parameters()).device


class Hook(object):
    def __init__(self, record_in=False, record_out=True, verbose=False):
        self.record_in = record_in
        self.record_out = record_out
        self.verbose = verbose

        if record_in:
            self.in_features = []
        if record_out:
            self.out_features = []

    def __call__(self, module, in_fea, out_fea):
        if self.verbose:
            print("hooker working", self)
        if self.record_in:
            self.in_features.append(in_fea)
        if self.record_out:
            self.out_features.append(out_fea)


def get_feature_maps(model,
                     inputs: tuple[torch.Tensor, torch.Tensor]) -> list[np.ndarray]:
    act_hook = Hook()
    model.eval()
    if type(model) == DDP:
        act_layer = model.module.act_layer
    else:
        act_layer = model.act_layer
    handle = act_layer.register_forward_hook(act_hook)

    with torch.no_grad():
        with autocast():
            model(inputs)
    feature_map = [fmap.detach().cpu().numpy() for fmap in act_hook.out_features]
    handle.remove()
    return feature_map


def get_gradients(model, inputs: torch.Tensor, target_classes: torch.Tensor) -> list[np.ndarray]:
    grad_hook = Hook(record_in=True, record_out=False)
    model.eval()
    if type(model) == DDP:
        act_layer = model.module.act_layer
    else:
        act_layer = model.act_layer
    handle = act_layer.register_full_backward_hook(grad_hook)

    with autocast():
        for input_tensor, target_cls in zip(inputs, target_classes):
            scores = model(input_tensor[None, :])  # (1, 1, H, W) -> (1, C)
            pred_cls = scores.argmax(dim=-1)
            if target_cls is None:
                target_cls = pred_cls  # (1, )
            else:
                assert target_cls < len(scores)
            scores[target_cls].backward()

    gradients = [grad[0].detach().cpu().numpy() for grad in grad_hook.in_features]
    handle.remove()
    gradients = [np.vstack(gradients[i::4]) for i in range(4)][::-1]
    return gradients


def similarity_matrix(data: torch.Tensor):
    N = data.shape[0]
    data = data.reshape(N, -1)

    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            matrix[i, j] = torch.cosine_similarity(data[i], data[j], dim=0).item()

    return matrix


def truncate_colormap(cmap_name: str, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
