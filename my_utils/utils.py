# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:39
# @Author  : WangYihao
# @File    : functions.py
import os
import time
from typing import Union, Optional

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from tqdm import tqdm

from my_utils.models import create_model, simple_Conv


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


def check_accuracy(test_model, loader, training=False, cls_num=4):
    confusion_matrix = np.zeros((cls_num,) * 2)
    device = get_device(test_model)
    test_model.eval()  # set model to evaluation mode
    iter_loader = iter(loader)
    tic = time.time()
    with torch.no_grad():
        for X, Y in iter_loader:
            X = X.to(device)  # move to device, e.g. GPU
            Y = Y.to(device)
            with autocast():
                _, preds = test_model((X, Y))
            for label, pred in zip(Y, preds):
                confusion_matrix[label, pred] += 1
    num_correct = confusion_matrix.trace()
    test_acc = float(num_correct) / confusion_matrix.sum()
    if training:
        return test_acc
    else:
        print(f"Test accuracy is : {100. * test_acc:.2f}%\tInfer time: {time.time() - tic}")
        return confusion_matrix


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


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


def load_model(run_name: str,
               log_root: str = '/home/wangyh/01-Projects/03-my/test_logs',
               ckpt_name: str = 'best') -> torch.nn.Module:
    """

    Args:
        run_name (str): distinguish a specific run, use run time for general cases.
        log_root (str):
        ckpt_name (str): 'best' or 'epoch_n', where 'n' is a multiple of 5.

    """
    print('Loading model parameters ...')
    run_dir = os.path.join(log_root, run_name)
    with open(os.path.join(run_dir, 'configs.yaml'), 'r') as stream:
        run_config = yaml.load(stream, Loader=yaml.FullLoader)

    checkpoint = torch.load(os.path.join(run_dir, f'checkpoints/{ckpt_name}.pth'))
    print('best val acc is:', checkpoint['best_val_acc'].item())

    model = create_model(**run_config['model_configs'])
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(new_state_dict)
    print('Successfully load model parameters!')

    return model


def fig2array(fig):
    from PIL import Image

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    image_array = Image.frombytes("RGBA", (w, h), buf.tostring())
    image_array = np.asarray(image_array)
    plt.close(fig)
    return image_array


def get_conv_weights(model: torch.nn.Module):
    kernels = []
    children = dict(model.named_children())
    if children == {}:
        if isinstance(model, nn.Conv2d):
            return [model.weight.detach().cpu().numpy()]
    else:
        for name, child in children.items():
            kernels.extend(get_conv_weights(child))
    return kernels


def get_feature_maps(model,
                     inputs: tuple[torch.Tensor, torch.Tensor]):
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


def get_avg_embeds(model, loader):
    device = get_device(model)

    embed_hook = Hook()
    model.eval()
    GAP_layer = model.GAP
    handle = GAP_layer.register_forward_hook(embed_hook)

    len_loader = len(loader)
    iter_loader = iter(loader)
    for _ in tqdm(range(len_loader)):
        images, lables = next(iter_loader)
        images = images.to(device)
        lables = lables.to(device)

        with torch.no_grad():
            with autocast():
                model((images, lables))
    embeds = [embed.detach().cpu().numpy().mean(0) for embed in embed_hook.out_features]
    handle.remove()

    return np.stack(embeds)
