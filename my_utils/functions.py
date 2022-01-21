# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:39
# @Author  : WangYihao
# @File    : functions.py


import torch
from torch import nn
import torch.nn.functional as F

import time


def Conv_BN_Relu(in_channel, out_channel, kernel_size=(3, 3), stride=None):
    if stride is not None:
        conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=False)
    else:
        conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding='same', bias=False)
    return nn.Sequential(
        conv_layer,
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


def save_model(model, model_type, optimizer, acc=00):
    model_paras = model.state_dict()
    print("Model parameters:")
    for k, v in model_paras.items():
        print(f"{k}:\t {v.size()}")

    optim_paras = optimizer.state_dict()
    print("\nOptimizer parameters:")
    for k, v in optim_paras.items():
        print(f"{k}")

    save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = f"saved_models/{acc}_polygen_{model_type}_{save_time}.pt"
    torch.save({
        "model_paras": model_paras,
        "optim_paras": optim_paras
    }, save_path)
    print(f"\nSuccessfully saved to {save_path}")
