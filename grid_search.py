# -*- coding: utf-8 -*-
# @Date    : 2022/3/14 23:49
# @Author  : WangYihao
# @File    : grid search.py
from my_utils.utils import train_a_model

for lr in [2e-4, 1e-4, 8e-5, 5e-5, 3e-5, 2e-5]:
    for batch_size in [64, 128, 256, 512]:
        for weight_decay in [0.05, 0.1, 0.15]:
            print(f'\n---- lr: {lr}, bs: {batch_size}, wd: {weight_decay} ----')
            train_config = {
                'dataset_dir': '/home/wangyh/01-Projects/03-my/Datasets/polygons_unfilled_32_2',
                'batch_size': batch_size,
                'device': 'cuda:7',
                'lr': lr,
                'cos_T': 10,
                'momentum': 0.9,
                'weight_decay': weight_decay,
            }
            train_a_model(configs=train_config)