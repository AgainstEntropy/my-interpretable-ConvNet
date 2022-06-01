# -*- coding: utf-8 -*-
# @Date    : 2022/4/19 08:58
# @Author  : WangYihao
# @File    : simcam.py
import numpy as np
import torch
from torchvision.transforms.functional import resize
from matplotlib import pyplot as plt

from .functional import adaptive_softmax, zoom_to_01
from .grad_cam import MyGradCAM


class MySimCAM(MyGradCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 use_cuda: bool = True):
        super().__init__(model, target_layer, use_cuda)

    def get_compose_weight(self) -> np.ndarray:
        # return (self.acts * self.input_tensor.cpu().numpy()).mean(axis=(-1, -2), keepdims=True)
        # return (self.acts * self.grads).mean(axis=(-1, -2), keepdims=True)
        return (self.acts * self.grads * self.input_tensor.cpu().numpy()).mean(axis=(-1, -2), keepdims=True)

        # return self.acts.mean(axis=(-1, -2), keepdims=True)

    def compose(self, compose_weights):
        return (self.acts * self.grads * compose_weights).sum(axis=1).squeeze()  # (1, C, H, W) -> (H, W)
