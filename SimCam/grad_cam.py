# -*- coding: utf-8 -*-
# @Date    : 2022/5/11 15:32
# @Author  : WangYihao
# @File    : GradCam.py
from typing import List

import numpy as np
import torch

from my_utils.utils import Hook, get_device


class MyGradCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 use_cuda: bool = True):
        self.target_cls = None
        self.use_cuda = use_cuda
        self.model = model.eval()
        self.target_layer = target_layer

        if not use_cuda:
            self.model = self.model.cpu()

        self._require_acts_and_grads()

    def compute_cam_map(self) -> np.ndarray:
        target_cls_score = self.forward()
        target_cls_score.backward()

        acts = self.act_hook.out_features[0].detach().cpu().numpy()
        grads = self.grad_hook.out_features[0][0].detach().cpu().numpy()

        self.remove_handles()

        compose_weights = self.get_compose_weight(acts, grads)
        null_cam_map = (acts * compose_weights).sum(axis=1)  # (N, C, H, W) -> (N, H, W)

        # normalize cam_map to [0, 1]
        normalized_cam_map = (null_cam_map - null_cam_map.min()) / (null_cam_map.max() - null_cam_map.min())

        return normalized_cam_map[0] if self.fig_num == 1 else normalized_cam_map

    def get_compose_weight(self,
                           acts: np.ndarray,
                           grads: np.ndarray) -> np.ndarray:
        return grads.mean(axis=(-1, -2), keepdims=True)

    def _require_acts_and_grads(self):
        self.act_hook = Hook()
        self.grad_hook = Hook()

        self.handles = []
        self.handles.append(self.target_layer.register_forward_hook(self.act_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(self.grad_hook))

    def remove_handles(self):
        for handle in self.handles:
            handle.remove()

    def forward(self):
        scores = self.model(self.input_tensor)
        if self.target_cls is None:
            self.target_cls = scores.argmax(dim=-1)
        else:
            assert self.target_cls < len(scores)
        return scores[self.target_cls]

    def __call__(self,
                 input_tensor: torch.Tensor,
                 target_cls: int = None) -> np.ndarray:
        self.input_tensor = input_tensor
        self.target_cls = target_cls
        if self.use_cuda:
            device = get_device(self.model)
            self.input_tensor = self.input_tensor.to(device)
        self.fig_num = self.input_tensor.size(0)
        return self.compute_cam_map()
