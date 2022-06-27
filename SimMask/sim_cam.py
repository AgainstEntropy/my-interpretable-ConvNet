# -*- coding: utf-8 -*-
# @Date    : 2022/4/19 08:58
# @Author  : WangYihao
# @File    : simcam.py
import numpy as np
import torch
from torchvision.transforms.functional import resize
from matplotlib import pyplot as plt

from .utils import get_device, get_feature_maps, get_gradients
from .functional import adaptive_softmax, zoom_to_01
from .grad_cam import MyGradCAM


class MySimCAM(MyGradCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 use_cuda: bool = True):
        super().__init__(model, target_layer, use_cuda)

    def get_compose_weight(self) -> np.ndarray:  # (1, C, 1, 1)
        # return (self.acts * self.input_tensor.cpu().numpy()).mean(axis=(-1, -2), keepdims=True)
        # return (self.acts * self.grads).mean(axis=(-1, -2), keepdims=True)
        return (self.acts * self.grads * self.input_tensor.cpu().numpy()).mean(axis=(-1, -2), keepdims=True)

        # return self.acts.mean(axis=(-1, -2), keepdims=True)

    def compose(self, compose_weights):
        return (self.acts * self.grads * compose_weights).sum(axis=1).squeeze()  # (1, C, H, W) -> (H, W)


class MyCAM4fmap(object):
    def __init__(self,
                 model: torch.nn.Module,
                 inputs: torch.Tensor,
                 target_classes: torch.Tensor):
        self.model = model.eval()
        self.inputs = inputs
        self.target_classes = target_classes

        device = get_device(self.model)
        self.inputs = self.inputs.to(device)
        self.target_classes = self.target_classes.to(device)

        self._get_ingredients()

    def _get_ingredients(self):
        self.feature_maps = get_feature_maps(self.model, self.inputs)
        self.gradients = get_gradients(self.model, self.inputs, self.target_classes)
        for fmap, grad in zip(self.feature_maps, self.gradients):
            assert fmap.shape == grad.shape

    def __call__(self, layer_index: int = -1) -> np.ndarray:
        self.layer_index = layer_index
        return self.compute_cam_map()

    def compute_cam_map(self):
        fmaps = self.feature_maps[self.layer_index]
        grads = self.gradients[self.layer_index]

        # Grad-CAM
        self.compose_weights = grads.mean(axis=(-1, -2), keepdims=True)  # (N, C, H, W) -> (N, C, 1, 1)
        # Sim-Mask
        # self.compose_weights = (fmaps * self.inputs.cpu().numpy()).mean(axis=(-1, -2), keepdims=True)

        cams = (self.compose_weights * fmaps).sum(axis=1).squeeze()  # (N, C, 1, 1)*(N, C, H, W) -> (N, H, W)
        # cams *= cams > 0

        # normalize cam_map to [0, 1]
        # norm_cams = np.zeros_like(cams)
        # for i, cam in enumerate(cams):
        #     norm_cams[i] = (cam - cam.min()) / (cam.max() - cam.min())

        return cams
