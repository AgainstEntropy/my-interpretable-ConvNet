# -*- coding: utf-8 -*-
# @Date    : 2022/5/11 15:32
# @Author  : WangYihao
# @File    : GradCam.py
from typing import List, Union, Tuple

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

    def compute_cam_map(self) -> Tuple[int, np.ndarray]:
        pred_cls, target_cls_score = self.forward()
        target_cls_score.backward()

        self.acts = self.act_hook.out_features[0].detach().cpu().numpy()
        self.grads = self.grad_hook.out_features[0][0].detach().cpu().numpy()

        compose_weights = self.get_compose_weight()
        null_cam_map = self.compose(compose_weights)

        # normalize cam_map to [0, 1]
        normalized_cam_map = (null_cam_map - null_cam_map.min()) / (null_cam_map.max() - null_cam_map.min())

        self.release()

        return pred_cls, normalized_cam_map

    def get_compose_weight(self) -> np.ndarray:
        return self.grads.mean(axis=(-1, -2), keepdims=True)

    def compose(self, compose_weights):
        return (self.acts * compose_weights).sum(axis=1).squeeze()  # (1, C, H, W) -> (H, W)

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
        scores = self.model(self.input_tensor)  # (1, C)
        pred_cls = scores.argmax(dim=-1)
        if self.target_cls is None:
            self.target_cls = pred_cls  # (1, )
        else:
            assert self.target_cls < len(scores)
        return pred_cls, scores[self.target_cls]

    def __call__(self,
                 input_tensor: torch.Tensor,
                 target_cls: int = None) -> np.ndarray:
        self.input_tensor = input_tensor
        self.target_cls = target_cls
        if self.use_cuda:
            device = get_device(self.model)
            self.input_tensor = self.input_tensor.to(device)
        self._require_acts_and_grads()
        return self.compute_cam_map()

    def release(self):
        self.model.zero_grad()
        self.remove_handles()
