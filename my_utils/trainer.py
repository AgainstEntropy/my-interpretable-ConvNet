# -*- coding: utf-8 -*-
# @Date    : 2022/4/30 15:52
# @Author  : WangYihao
# @File    : trainer.py

import os
import platform
import random
import time
from decimal import Decimal

import numpy as np
import wandb
import yaml
from loguru import logger
from prettytable import PrettyTable
from tqdm import tqdm
from fairscale.optim.oss import OSS
import torch
from torch import optim, nn, distributed
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

from my_utils import data
from my_utils.models import create_model
from my_utils.utils import AverageMeter, correct_rate, Hook, get_conv_weights
from my_utils.vis import vis_4D

cudnn.benchmark = True


def seed_worker(worker_id):
    # print(torch.initial_seed())
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _set_seed(seed, deterministic=False):
    """
    seed manually to make runs reproducible
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option
        for CUDNN backend
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


class Trainer(object):
    def __init__(self, cfg):
        tic = time.time()
        self.cfg = cfg
        self.model_cfgs = cfg['model_configs']
        self.train_cfgs = cfg['train_configs']
        self.dataset_cfgs = cfg['dataset_configs']
        self.loader_kwargs = cfg['loader_kwargs']
        self.optim_kwargs = cfg['optim_kwargs']
        self.schedule_cfgs = cfg['schedule_configs']
        self.dist_cfgs = cfg['distributed_configs']
        self.log_cfgs = cfg['log_configs']

        if self.dist_cfgs['distributed']:
            distributed.init_process_group(backend='nccl',
                                           init_method='tcp://127.0.0.1:' + self.dist_cfgs['port'],
                                           world_size=self.dist_cfgs['world_size'],
                                           rank=self.dist_cfgs['local_rank'])
        _set_seed(self.train_cfgs['seed'] + self.dist_cfgs['local_rank'], deterministic=True)
        if torch.cuda.is_available():
            self.device = f'cuda:{self.dist_cfgs["local_rank"]}'
        else:
            self.device = "cpu"
        self.dist_cfgs['device'] = self.device

        save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        run_dir = os.path.join(os.getcwd(),
                               f"{self.log_cfgs['log_dir']}/"
                               f"{self.model_cfgs['type']}/"
                               f"KS_{self.model_cfgs['kernel_size']}_"
                               f"ACT_{self.model_cfgs['act']}_"
                               f"Norm_{self.model_cfgs['norm']}")
        self.log_dir = os.path.join(run_dir, save_time)
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.dist_cfgs['local_rank'] == 0:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            with open(os.path.join(self.log_dir, 'configs.yaml'), 'w', encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, default_flow_style=False, allow_unicode=True)

        self.start_epoch = 0
        self.steps = 0
        self.epoch = 0
        self.min_loss = float('inf')
        self.val_best_acc_total = 0.0
        self.val_metrics = {'current_acc': 0.0, 'best_acc': 0.0,
                            'best_epoch': 0}

        self._build_model()
        if self.dist_cfgs['distributed']:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.dist_cfgs['local_rank']],
                             output_device=self.dist_cfgs['local_rank'],
                             find_unused_parameters=False)

        if self.train_cfgs['mode'] == 'train':
            self.train_loader, self.train_sampler = self._load_dataset(phase='train')
            self.val_loader, self.val_sampler = self._load_dataset(phase='val')
        if self.train_cfgs['mode'] == 'test':
            self.test_loader, self.test_sampler = self._load_dataset(phase='test')

        self._load_optimizer()

        if self.train_cfgs['resume']:
            checkpoint_path = self.train_cfgs['resume_path']
            assert os.path.exists(checkpoint_path)
            self.load_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            self._init_wandb(project_name='my', run_name=save_time)
            self.vis_loader, _ = self._load_dataset(phase='vis')

        if self.dist_cfgs['local_rank'] == 0:
            print(f"{time.time() - tic} sec are used to initialize a Trainer.")

    def _init_wandb(self, project_name, run_name):
        wandb_config = {
            "kernel_size": self.model_cfgs['kernel_size'],
            "depths": self.model_cfgs['depths'],
            "dims": self.model_cfgs['dims'],
            "activation": self.model_cfgs['act'],
            "normalization": self.model_cfgs['norm'],
            "use_GSP": self.model_cfgs['use_GSP'],
            "batch_size": self.train_cfgs['batch_size'],
            "lr_backbone": self.optim_kwargs['lr'],
            "optimizer": self.optim_kwargs['optimizer'],
            "weight_decay": self.optim_kwargs['weight_decay'],
            "epochs": self.schedule_cfgs['max_epoch'],
        }
        wandb.init(project=project_name, entity="against-entropy", name=run_name, dir=self.log_dir,
                   config=wandb_config)
        wandb.watch(self.model)

        config_table = PrettyTable()
        config_table.add_column('Phase', list(wandb_config))
        config_table.add_column('Val', list(wandb_config.values()))

        logger.info('\n' + config_table.get_string())

    def _build_model(self):
        self.model = create_model(**self.model_cfgs)
        self.model.to(self.device)

    def _load_dataset(self, phase='train'):
        trans = T.Compose([
            T.ToTensor(),
            T.Resize((self.dataset_cfgs['fig_resize'],) * 2)
        ])
        if self.dataset_cfgs['preprocess']:
            trans = T.Compose([trans, T.Normalize(self.dataset_cfgs['mean'], self.dataset_cfgs['std'])])
        dataset = data.make_dataset(
            phase=phase,
            dataset_dir=self.train_cfgs['dataset_dir'],
            transform=trans
        )

        sampler = DistributedSampler(dataset, shuffle=True) \
            if self.dist_cfgs['distributed'] else None
        data_loader = DataLoader(dataset,
                                 sampler=None if phase == 'vis' else sampler,
                                 worker_init_fn=seed_worker,
                                 shuffle=(sampler is None or phase == 'vis'),
                                 drop_last=(phase == 'train'),
                                 **self.loader_kwargs)

        return data_loader, sampler

    def _load_optimizer(self):
        base_optimizer = None
        optim_type = self.optim_kwargs.pop('optimizer')
        if optim_type == 'SGD':
            base_optimizer = optim.SGD
            self.optim_kwargs['momentum'] = 0.9
        elif optim_type == 'Adam':
            base_optimizer = optim.Adam
            self.optim_kwargs['betas'] = (0.9, 0.999)
        elif optim_type == 'AdamW':
            base_optimizer = optim.AdamW
            self.optim_kwargs['betas'] = (0.9, 0.999)
        else:
            print(f"{optim_type} not support.")
            exit(0)

        if self.dist_cfgs['distributed']:
            # Wrap a base optimizer into OSS
            self.optimizer = OSS(
                params=self.model.parameters(),
                optim=base_optimizer,
                **self.optim_kwargs,
            )
        else:
            self.optimizer = base_optimizer(
                params=self.model.parameters(),
                **self.optim_kwargs,
            )

        if self.schedule_cfgs['schedule_type'] == 'cosine_warm':
            self.schedule_cfgs['max_epoch'] = \
                int((self.schedule_cfgs['cos_mul'] ** self.schedule_cfgs['cos_iters'] - 1) / \
                    (self.schedule_cfgs['cos_mul'] - 1) * self.schedule_cfgs['cos_T'])
            self.scheduler = \
                optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                               T_0=self.schedule_cfgs['cos_T'], T_mult=2)

        if self.train_cfgs['amp']:
            self.scaler = GradScaler()

        self.optim_kwargs['optimizer'] = optim_type

    def run(self):
        for epoch in range(self.start_epoch, self.schedule_cfgs['max_epoch']):

            if self.dist_cfgs['distributed']:
                self.train_sampler.set_epoch(epoch)

            train_loss, train_acc = self.train(epoch)
            self.min_loss = min(self.min_loss, train_loss)

            val_loss, val_acc = self.val(epoch)
            self.epoch += 1

            if self.dist_cfgs['local_rank'] == 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(tag=f'optimizer/lr_group_{i}',
                                           scalar_value=param_group['lr'],
                                           global_step=epoch)
                    wandb.log({f"optimizer/lr_group_{i}": param_group['lr']})

                self.writer.add_scalars('Metric/acc', {'train': train_acc, 'val': val_acc}, epoch + 1)
                self.writer.add_scalars('Metric/loss', {'train': train_loss, 'val': val_loss}, epoch + 1)

                wandb.log({
                    'Metric/acc/train': train_acc,
                    'Metric/acc/val': val_acc,
                    'Metric/acc/best_acc': self.val_metrics['best_acc'],
                    'Metric/loss/train': train_loss,
                    'Metric/loss/val': val_loss
                })

                self.log_images()

            self.scheduler.step()

            if ((epoch + 1) % self.log_cfgs['save_epoch_interval'] == 0) \
                    or (epoch + 1) == self.schedule_cfgs['max_epoch']:
                checkpoint_path = os.path.join(self.ckpt_dir, f"epoch_{(epoch + 1)}.pth")
                self.save_checkpoint(checkpoint_path)

        if self.dist_cfgs['distributed']:
            distributed.destroy_process_group()

        if self.dist_cfgs['local_rank'] == 0:
            wandb.finish()

    def train(self, epoch):
        self.model.train()
        len_loader = len(self.train_loader)
        iter_loader = iter(self.train_loader)

        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()

        pbar = None
        if self.dist_cfgs['local_rank'] == 0:
            pbar = tqdm(total=len_loader,
                        dynamic_ncols=True,
                        ascii=(platform.version() == 'Windows'))

        for step in range(len_loader):
            try:
                inputs, labels = next(iter_loader)
            except Exception as e:
                logger.critical(e)
                continue

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            if self.train_cfgs['amp']:
                with autocast():
                    loss, preds = self.model((inputs, labels))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, preds = self.model((inputs, labels))
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1
            loss = loss.detach().clone()
            acc_recorder.update(correct_rate(preds, labels), batch_size)

            if self.dist_cfgs['distributed']:
                distributed.reduce(loss, 0)
                loss /= self.dist_cfgs['world_size']
            loss_recorder.update(loss.item(), batch_size)

            if self.dist_cfgs['local_rank'] == 0:
                last_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
                last_lr_string = "lr " + ' '.join(f"{Decimal(lr):.1E}" for lr in last_lr)

                pbar.set_description(
                    f"train epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                    f"Iter {self.steps}/{len_loader * self.schedule_cfgs['max_epoch']}  "
                    f"{last_lr_string}  "
                    f"----  "
                    f"loss {loss_recorder.avg:.4f}  "
                    f"top1_acc {acc_recorder.avg:.2%}")
                pbar.update()

                if self.steps % self.log_cfgs['snapshot_interval'] == 0:
                    checkpoint_path = os.path.join(self.ckpt_dir, "latest.pth")
                    self.save_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            pbar.close()

            logger.info(
                f"train epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                f"Iter {self.steps}/{len_loader * self.schedule_cfgs['max_epoch']}  "
                f"----  "
                f"loss {loss_recorder.avg:.4f} "
                f"top1_acc {acc_recorder.avg:.2%}")

        return loss_recorder.avg, acc_recorder.avg

    def val(self, epoch):
        self.model.eval()
        len_loader = len(self.val_loader)
        iter_loader = iter(self.val_loader)

        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()

        pbar = None
        if self.dist_cfgs['local_rank'] == 0:
            pbar = tqdm(total=len_loader,
                        dynamic_ncols=True,
                        ascii=(platform.version() == 'Windows'))

        for step in range(len_loader):
            try:
                inputs, labels = next(iter_loader)
            except Exception as e:
                logger.critical(e)
                continue
            if not self.dist_cfgs['distributed'] and self.dist_cfgs['local_rank'] == 0 \
                    and epoch % 10 == 0 and step == len_loader - 1:
                inputs_collector = inputs.clone()
                labels_collector = labels.clone()
                GAP_hook = Hook()
                hook_handle = self.model.GAP.register_forward_hook(GAP_hook)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            with torch.no_grad():
                if self.train_cfgs['amp']:
                    with autocast():
                        loss, preds = self.model((inputs, labels))
                else:
                    loss, preds = self.model((inputs, labels))

            loss = loss.detach().clone()
            acc_recorder.update(correct_rate(preds, labels), batch_size)

            if self.dist_cfgs['distributed']:
                distributed.reduce(loss, 0)
                loss /= self.dist_cfgs['world_size']
            loss_recorder.update(loss.item(), batch_size)

            if self.dist_cfgs['local_rank'] == 0:
                pbar.set_description(
                    f"val epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                    f"Step {step}/{len_loader}  "
                    f"------  "
                    f"loss {loss_recorder.avg:.4f}  "
                    f"top1_acc {acc_recorder.avg:.2%}")
                pbar.update()

        if self.dist_cfgs['local_rank'] == 0:
            if not self.dist_cfgs['distributed'] and epoch % 10 == 0:
                embed_features = torch.cat(GAP_hook.out_features, dim=0)
                self.writer.add_embedding(mat=embed_features,
                                          metadata=labels_collector,
                                          label_img=inputs_collector,
                                          global_step=epoch,
                                          tag='gap_embed')

                hook_handle.remove()
                del GAP_hook
            pbar.close()

            logger.info(
                f"val epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                f"------  "
                f"loss {loss_recorder.avg:.4f}  "
                f"top1_acc {acc_recorder.avg:.2%}")

            self.val_metrics['current_acc'] = acc_recorder.avg
            if acc_recorder.avg > self.val_metrics['best_acc']:
                self.val_metrics['best_acc'] = acc_recorder.avg
                self.val_metrics['best_epoch'] = epoch + 1

                checkpoint_path = os.path.join(self.ckpt_dir, "best.pth")
                self.save_checkpoint(checkpoint_path)

            res_table = PrettyTable()
            res_table.add_column('Phase', ['Current Acc', 'Best Acc', 'Best Epoch'])
            res_table.add_column('Val', [f"{self.val_metrics['current_acc']:.2%}",
                                         f"{self.val_metrics['best_acc']:.2%}",
                                         self.val_metrics['best_epoch']])

            logger.info(f'Performance on validation set at epoch: {epoch + 1}')
            logger.info('\n' + res_table.get_string())

        return loss_recorder.avg, acc_recorder.avg

    def log_images(self):
        conv_weights = get_conv_weights(self.model)
        for idx, weight in enumerate(conv_weights):
            fig_array = vis_4D(data=weight, figsize_factor=3,
                               cmap='viridis', return_fig_array=True)
            wandb.log({'kernels': wandb.Image(fig_array, caption=f"idx:{idx}  epoch:{self.epoch}")})

    def save_checkpoint(self, path):
        # self.optimizer.consolidate_state_dict()
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        if self.dist_cfgs['local_rank'] == 0:
            save_dict = {
                'model': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'iteration': self.steps,
                'best_val_acc': self.val_metrics['best_acc'],
                'best_epoch': self.val_metrics['best_epoch'],
                'val_best_acc_total': self.val_best_acc_total,
            }
            torch.save(save_dict, path)

    def load_checkpoint(self, path):
        ckpt = None
        if self.dist_cfgs['local_rank'] == 0:
            ckpt = torch.load(path, map_location={'cuda:0': f'cuda:{self.dist_cfgs["local_rank"]}'})
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch']
        self.steps = ckpt['iteration']
        self.val_metrics['best_epoch'] = ckpt['best_epoch']
        self.val_metrics['best_acc'] = ckpt['best_val_acc']
        self.val_best_acc_total = ckpt['val_best_acc_total']
