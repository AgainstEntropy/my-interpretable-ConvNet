# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 14:47
# @Author  : WangYihao
# @File    : data.py

import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image


class MyDataset(Dataset):
    def __init__(self, ROOT, transform=None, target_transform=None):
        self.data = []
        for idx, path in enumerate(sorted(os.listdir(ROOT))):
            class_path = os.path.join(ROOT, path)
            class_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            self.data.extend(list(zip(class_files, [idx] * len(class_files))))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_dir, label = self.data[index]
        img = read_image(img_dir).numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)


def make_datasets(dataset_dir, loader_kwargs, transform):
    train_data = MyDataset(os.path.join(dataset_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_data, **loader_kwargs)
    val_data = MyDataset(os.path.join(dataset_dir, 'val'), transform=transform)
    val_loader = DataLoader(val_data, **loader_kwargs)
    test_data = MyDataset(os.path.join(dataset_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_data, **loader_kwargs)

    sample_step = 10
    small_train_data = Subset(train_data, torch.arange(0, len(train_data) - 1, sample_step))
    small_train_loader = DataLoader(small_train_data, **loader_kwargs)

    check_loaders = {'train': small_train_loader,
                     'val': val_loader}

    return train_loader, test_loader, check_loaders
