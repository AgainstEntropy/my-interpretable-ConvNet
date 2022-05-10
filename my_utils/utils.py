# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:39
# @Author  : WangYihao
# @File    : functions.py
import os
import time

import adabound
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from openpyxl import load_workbook

from my_utils import data, models
from my_utils.models import create_model


class Hook(object):
    def __init__(self):
        self.features_in = []
        self.features_out = []

    def __call__(self, module, fea_in, fea_out):
        print("hooker working", self)
        self.features_in.append(fea_in)
        self.features_out.append(fea_out)


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


def sim(imgs_act, img):
    """

    Args:
        imgs_act ():
        img ():

    Returns: Similarity scores between activations and original image.

    """
    assert len(imgs_act.size()) >= 3
    chans = imgs_act.size(0)
    if len(img.size() >= 3):
        img = img.squeeze()
    sims = torch.zeros(chans)
    for chan, act in enumerate(imgs_act):
        sims[chan] = (act * img).sum() / \
                     torch.sqrt(torch.sum(act ** 2) * torch.sum(img ** 2))
    return torch.softmax(sims, dim=0)


def save_model(model, optimizer, scheduler, save_dir, acc=00):
    model_paras = model.state_dict()
    optim_paras = optimizer.state_dict()
    scheduler_main_paras = scheduler.state_dict()

    save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = os.path.join(save_dir, f'{acc:.1f}_{save_time}.pt')
    torch.save({
        "model_paras": model_paras,
        "optim_paras": optim_paras,
        "scheduler_paras": scheduler_main_paras
    }, save_path)

    print(f"\nSuccessfully saved model, optimizer and scheduler to {save_path}")


def get_device(model):
    return next(model.parameters()).device


def check_accuracy(test_model, loader, training=False):
    num_correct = 0
    num_samples = 0
    device = get_device(test_model)
    test_model.eval()  # set model to evaluation mode
    tic = time.time()
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(loader):
            X = X.to(device, dtype=torch.float32)  # move to device, e.g. GPU
            Y = Y.to(device, dtype=torch.int)
            scores = test_model(X)
            num_correct += (scores.argmax(axis=1) == Y).sum()
            num_samples += len(scores)
    test_acc = float(num_correct) / num_samples
    if training:
        return test_acc
    else:
        print(f"Test accuracy is : {100. * test_acc:.2f}%\tInfer time: {time.time() - tic}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = -1
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_rate(preds, labels):
    assert len(preds) == len(labels)
    num_correct = (preds == labels).sum()
    return num_correct / len(preds)
