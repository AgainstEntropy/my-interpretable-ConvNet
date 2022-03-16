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

from my_utils import data, models
from my_utils.models import create_model


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


def save_model(model, optimizer, scheduler, model_type, save_dir, acc=00):
    model_paras = model.state_dict()
    optim_paras = optimizer.state_dict()
    scheduler_main_paras = scheduler.state_dict()

    save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = os.path.join(save_dir, f'{acc}_{model_type}_{save_time}.pt')
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


def trainer(model, optimizer, scheduler, loss_fn, train_loader,
            check_fn, check_loaders, batch_step, epochs=2, log_every=10, writer=None):
    """

    Args:
        batch_step (int):
        epochs (int):
        log_every (int): log info per log_every batches.
        writer :

    Returns:
        batch_step (int):
    """
    device = get_device(model)
    # batch_size = train_loader.batch_size
    check_loader_train = check_loaders['train']
    check_loader_val = check_loaders['val']
    iters = len(train_loader)
    for epoch in range(1, epochs + 1):
        tic = time.time()
        for batch_idx, (X, Y) in enumerate(train_loader):
            batch_step += 1
            model.train()
            X = X.to(device, dtype=torch.float32)
            Y = Y.to(device, dtype=torch.int64)
            # print(X.device, model.device)
            scores = model(X)
            loss = loss_fn(scores, Y)
            if writer is not None:
                writer.add_scalar('Metric/loss', loss.item(), batch_step)
                writer.add_scalar('Hpara/lr', optimizer.param_groups[0]['lr'], batch_step)

            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(batch_step / iters)

            # check accuracy
            if batch_idx % log_every == 0:
                model.eval()
                train_acc = check_fn(model, check_loader_train, training=True)
                val_acc = check_fn(model, check_loader_val, training=True)
                if writer is not None:
                    writer.add_scalars('Metric/acc', {'train': train_acc, 'val': val_acc}, batch_step)
                print(f'Epoch: {epoch} [{batch_idx}/{iters}]\tLoss: {loss:.4f}\t'
                      f'Val acc: {100. * val_acc:.1f}%')

        print(f'====> Epoch: {epoch}\tTime: {time.time() - tic}s')

    return batch_step


def train_a_model(model_configs=None, train_configs=None, loader_kwargs=None):
    """
        Train a model from zero.
    """
    if train_configs is None:
        train_configs = {
            'log_dir': 'newruns',
            'dataset_dir': '/home/wangyh/01-Projects/03-my/Datasets/polygons_unfilled_64_3',
            'batch_size': 256,
            'epochs': 50,
            'device': 'cuda:7',
            'optim': 'Adam',
            'lr': 1e-4,
            'schedule': 'cosine_warm',
            'cos_T': 15,
            'cos_mul': 2,
            'cos_iters': 3,
            'momentum': 0.9,
            'weight_decay': 0.05,
        }
    # make dataset
    fig_resize = 64
    # mean, std = torch.tensor(0.2036), torch.tensor(0.4027)  # polygons_unfilled_32_2
    mean, std = torch.tensor(0.1094), torch.tensor(0.3660)  # polygons_unfilled_64_3
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((fig_resize, fig_resize)),
        transforms.Normalize(mean, std)
    ])
    if loader_kwargs is None:
        loader_kwargs = {
            'batch_size': train_configs['batch_size'],  # default:1
            'shuffle': True,  # default:False
            'num_workers': 4,  # default:0
            'pin_memory': True,  # default:False
            'drop_last': True,  # default:False
            'prefetch_factor': 4,  # default:2
            'persistent_workers': False  # default:False
        }
    train_loader, test_loader, check_loaders = data.make_datasets(
        dataset_dir=train_configs['dataset_dir'],
        loader_kwargs=loader_kwargs,
        transform=T
    )

    # create model
    if model_configs is None:
        model_configs = {
            'type': 'simple_conv',
            'kernel_size': 3,
            'depths': (1, 1, 1),
            'dims': (4, 8, 16)
        }
    model, optimizer, scheduler = [None] * 3
    # define model
    model = create_model(**model_configs)
    model = model.to(train_configs['device'])

    # define optimizer
    if train_configs['optim'] == 'Adam':
        optimizer = optim.Adam(params=[{'params': model.parameters(), 'initial_lr': train_configs['lr']}],
                               lr=train_configs['lr'],
                               weight_decay=train_configs['weight_decay'])
    elif train_configs['optim'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=train_configs['lr'],
                                weight_decay=train_configs['weight_decay'])
    elif train_configs['optim'] == 'AdaBound':
        optimizer = adabound.AdaBound(model.parameters(),
                                      lr=train_configs['lr'],
                                      weight_decay=train_configs['weight_decay'],
                                      final_lr=0.1)
    elif train_configs['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=train_configs['lr'],
                              weight_decay=train_configs['weight_decay'],
                              momentum=train_configs['momentum'])

    # define lr scheduler
    if train_configs['schedule'] == 'cosine_warm':
        train_configs['epochs'] = int((train_configs['cos_mul'] ** train_configs['cos_iters'] - 1) / \
                                      (train_configs['cos_mul'] - 1) * train_configs['cos_T'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=train_configs['cos_T'], T_mult=2)
    elif train_configs['schedule'] == 'cosine_anneal':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_configs['cos_T'])
    loss_func = nn.CrossEntropyLoss()
    print(f"model ({model_configs['type']}) is on {next(model.parameters()).device}")

    # tensorboard writer
    save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    log_dir = os.path.join(os.getcwd(),
                           f"{train_configs['log_dir']}/{model_configs['type']}/"
                           f"KS_{model_configs['kernel_size']}_"
                           f"LR_{train_configs['lr']:.1e}_"
                           f"WD_{train_configs['weight_decay']:.1e}")
    log_dir = os.path.join(log_dir, save_time)
    writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, 'para.txt'), mode='w') as f:
        f.write('## -- model configs -- ##\n')
        for k, v in model_configs.items():
            f.write(f'{k} :\t{v}\n')

        f.write('\n## -- dataset configs -- ##')
        for k, v in loader_kwargs.items():
            f.write(f'{k} :\t{v}\n')

        f.write('\n## -- train configs -- ##')
        for k, v in train_configs.items():
            f.write(f'{k} :\t{v}\n')

    trainer(model=model, optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_func,
            train_loader=train_loader,
            check_fn=check_accuracy,
            check_loaders=check_loaders,
            batch_step=0, epochs=train_configs['epochs'], log_every=40000 // train_configs['batch_size'] // 4,
            writer=writer)

    writer.close()

    final_acc = check_accuracy(model, test_loader, True)
    save_model(model, optimizer, scheduler,
               model_configs['type'], log_dir, acc=int(100 * final_acc))
