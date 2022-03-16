# -*- coding: utf-8 -*-
# @Date    : 2022/3/14 23:49
# @Author  : WangYihao
# @File    : fine_tune.py

from my_utils.utils import train_a_model

model_configs = {
    'type': 'simple_conv',
    'kernel_size': 5,
    'depths': (1, 1, 1),
    'dims': (4, 8, 16)
}
train_configs = {
    'log_dir': 'newruns',
    'dataset_dir': '/home/wangyh/01-Projects/03-my/Datasets/polygons_unfilled_64_3',
    'batch_size': 512,
    'epochs': 140,
    'device': 'cuda:6',
    'optim': 'AdamW',
    'lr': 1.5e-4,
    'schedule': 'cosine_warm',
    'cos_T': 40,
    'cos_mul': 2,
    'cos_iters': 2,
    'momentum': 0.9,
    'weight_decay': 0.005,
}
loader_kwargs = {
    'batch_size': train_configs['batch_size'],  # default:1
    'shuffle': True,  # default:False
    'num_workers': 4,  # default:0
    'pin_memory': True,  # default:False
    'drop_last': True,  # default:False
    'prefetch_factor': 4,  # default:2
    'persistent_workers': False  # default:False
}

train_a_model(model_configs=model_configs, train_configs=train_configs, loader_kwargs=loader_kwargs)
