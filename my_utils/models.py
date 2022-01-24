# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:37
# @Author  : WangYihao
# @File    : model.py

import torch
from torch import nn
import torch.nn.functional as F


# from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)  # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)  # pointwise/1x1 convs, implemented with linear layers
        # self.act = nn.GELU()
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # x = input_x + self.drop_path(x)

        return x


class my_ConvNeXt(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 1
        num_classes (int): Number of classes for classification head. Default: 5
        depths (tuple(int)): Number of blocks at each stage. Default: [1, 1, 1, 1]
        dims (int): Feature dimension at each stage. Default: [8, 16, 32]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, num_classes=4,
                 depths=[1, 1, 1], dims=[8, 16, 32], drop_path_rate=0.,
                 layer_scale_init_value=1e-2, head_init_scale=1.,
                 ):
        super().__init__()

        self.num_layers = len(dims)
        # self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #     # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        #     nn.BatchNorm2d(dims[0])
        # )
        # self.downsample_layers.append(stem)
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #         # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #         nn.BatchNorm2d(dims[i]),
        #         nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
        #     )
        #     self.downsample_layers.append(downsample_layer)

        self.connection_layers = nn.ModuleList()  # pointwise/1x1 convs layers between different stages, implemented with linear layers
        start = nn.Linear(in_chans, dims[0])
        self.connection_layers.append(start)
        for i in range(self.num_layers - 1):
            pw_layer = nn.Linear(dims[i], dims[i + 1])
            self.connection_layers.append(pw_layer)

        self.norm_layers = nn.ModuleList()  # normalization layers between different stages
        norm_layer = nn.BatchNorm2d(dims[0])
        self.norm_layers.append(norm_layer)
        for i in range(self.num_layers - 1):
            norm_layer = nn.BatchNorm2d(dims[i])
            self.norm_layers.append(norm_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # cur = 0
        for i in range(self.num_layers):
            # stage = nn.Sequential(
            #     *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
            #             layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            # )
            stage = nn.Sequential(
                *[Block(dim=dims[i], layer_scale_init_value=layer_scale_init_value) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            # cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.norm = nn.BatchNorm1d(dims[-1])  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_connection(self, x, block_idx=0):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.connection_layers[block_idx](x)  # (N, H, W, C[i]) -> (N, H, W, C[i+1])
        return x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    def forward_1st_block(self, x):
        x = self.forward_connection(x)
        x = self.norm_layers[0](x)
        return self.stages[0](x)

    def forward_block(self, x, block_idx):
        x = self.norm_layers[block_idx](x)
        x = self.forward_connection(x, block_idx)
        return self.stages[block_idx](x)

    def forward_features(self, x):
        x = self.forward_1st_block(x)
        for block_idx in range(1, self.num_layers):
            # x = self.downsample_layers[i](x)
            x = self.forward_block(x, block_idx)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class simple_Conv(nn.Module):
    def __init__(self, in_chans=1, num_classes=4,
                 depths=[1, 1, 1], dims=[1, 1, 1]):
        super().__init__()

        self.blocks = nn.ModuleList()
        # TODO: complete simple_conv
        start = nn.Linear(in_chans, dims[0])
        self.connection_layers.append(start)
        for i in range(self.num_layers - 1):
            pw_layer = nn.Linear(dims[i], dims[i + 1])
            self.connection_layers.append(pw_layer)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x1 = self.conv_layer1(x)
        x2 = self.conv_layer2(x1)
        x3 = self.conv_layer3(x2)
        x4 = x3.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
        scores = self.head(x4)

        return scores


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
