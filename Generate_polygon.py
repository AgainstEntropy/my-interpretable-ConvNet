# -*- coding: utf-8 -*-
# @Date    : 2022/1/8 13:25
# @Author  : WangYihao
# @File    : Generate_polygon.py

import os
import argparse
import random

import numpy as np
import cv2
from tqdm import tqdm


def angles(n, rr):
    angle_list = np.zeros(n)
    angle_n = 2 * np.pi / n
    angle_list[0] = random.uniform(0, angle_n)
    for i in range(1, n):
        angle_list[i] = angle_list[i - 1] + angle_n + np.pi / 180 * random.uniform(-rr, rr)

    return angle_list


def points(n, width=32, r=13):
    angle_list = angles(n, 20)
    center = np.array((width // 2,) * 2)
    cs_list = np.vstack((np.cos(angle_list), -np.sin(angle_list)))
    point_list = center + r * cs_list.T

    return np.array(point_list, dtype=int)


def draw_polygon(point_list, fill=False, width=32):
    img = np.zeros((width, width))
    if fill:
        cv2.fillPoly(img, [point_list], color=255)
    else:
        cv2.polylines(img, [point_list], isClosed=True, color=255, thickness=1)

    return img


parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, required=True, default='train',
                    help="the dataset you want to generate.")
parser.add_argument('-f', '--fill', default=False,
                    help="whether or not to fill the polygons.")
parser.add_argument('-fn', '--figNum', type=int, default=10000,
                    help="the number of pictures to generate.")
parser.add_argument('-w', '--width', type=int, default=32,
                    help="the width of figure to generate.")
parser.add_argument('-an', '--angNums', type=list, default=list(range(3, 7)),
                    help="how many angles to be contained in one figure, so as the species of polygons.")
args = parser.parse_args()

if __name__ == '__main__':
    fill = args.fill
    dataset = args.dataset
    figNum = args.figNum
    angNums = args.angNums
    width = args.width
    if fill:
        ROOT = f'Datasets/polygons_filled_{width}/'
    else:
        ROOT = f'Datasets/polygons_unfilled_{width}/'
    ROOT += dataset
    for angNum in tqdm(angNums):
        Path = os.path.join(ROOT, f'{angNum}')
        if not os.path.isdir(Path):
            os.makedirs(Path)
        for j in tqdm(range(figNum)):
            random.seed()
            point_list = points(angNum, width, r=width * 0.4)
            img = draw_polygon(point_list, fill, width)
            img_address = os.path.join(Path, f"{angNum}_{j}.png")
            cv2.imwrite(img_address, img)

    # point_list = points(n=4, width=width, r=13)
    # img = draw_polygon(point_list)
    # plt.imshow(img, 'gray')
    # plt.show()
