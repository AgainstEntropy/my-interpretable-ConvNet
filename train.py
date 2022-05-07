# -*- coding: utf-8 -*-
# @Date    : 2022/3/14 23:49
# @Author  : WangYihao
# @File    : train.py
import argparse
import os

import torch
import torch.multiprocessing as mp
import yaml


def main(cfg):
    loader_cfgs = cfg['loader_kwargs']
    train_cfgs = cfg['train_configs']
    dist_cfgs = cfg['distributed_configs']
    log_cfgs = cfg['log_configs']

    os.makedirs(log_cfgs['log_dir'], exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = dist_cfgs['device_ids']

    world_size = len(dist_cfgs['device_ids'].split(','))
    dist_cfgs['distributed'] = True if world_size > 1 else False
    dist_cfgs['world_size'] = world_size
    loader_cfgs['batch_size'] = train_cfgs['batch_size'] // world_size

    if dist_cfgs['distributed']:
        print(f"Using devices: {dist_cfgs['device_ids']}")
        mp.spawn(worker, nprocs=world_size, args=(cfg,))
    else:
        worker(0, cfg)


def worker(rank, cfg):
    torch.cuda.set_device(rank)
    cfg['distributed_configs']['local_rank'] = rank
    from my_utils.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training a classifier convnet")
    parser.add_argument('-cfg', '--config', type=str, default='configs/default.yaml')
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-r', '--resume', action='store_true', help='load previously saved checkpoint')
    parser.add_argument('-log', '--log_dir', type=str, default='test_runs', help='where to log train results')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('-g', '--gpu_ids', type=lambda x: x.replace(" ", ""), default='0,1', help='available gpu ids')
    parser.add_argument('--port', type=str, default='4250', help='port number of distributed init')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config['train_configs']['batch_size'] = args.batch_size
    config['train_configs']['resume'] = args.resume

    config['optim_kwargs']['lr'] = args.learning_rate

    config['distributed_configs']['device_ids'] = args.gpu_ids
    config['distributed_configs']['port'] = args.port

    config['log_configs']['log_dir'] = args.log_dir

    main(config)
