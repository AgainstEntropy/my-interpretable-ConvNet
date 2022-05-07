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


def make_dataset(phase, dataset_dir, transform):
    return MyDataset(
        ROOT=os.path.join(dataset_dir, phase),
        transform=transform
    )
