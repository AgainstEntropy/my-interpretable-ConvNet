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


def points(n, width=64, float_rate=(0.05, 0.05)):
    angle_list = angles(n, 20)
    w2 = width / 2
    c_float_rate, r_float_rate = float_rate
    center = np.array((w2,) * 2) + c_float_rate * w2 * np.random.randn(2)
    cs_list = np.vstack((np.cos(angle_list), -np.sin(angle_list)))
    r = w2 * (1 - sum(float_rate))
    point_list = center + r * cs_list.T + r_float_rate * w2 * np.random.randn(2)

    return r, np.array(point_list, dtype=int)


def get_mPoints_and_eLength(point_list):
    mPoints = np.zeros_like(point_list)
    eLengths = np.zeros(point_list.shape[0])
    for i, p in enumerate(point_list):
        mPoints[i] = (point_list[i - 1] + p) / 2
        eLengths[i] = np.linalg.norm(point_list[i - 1] - p)

    return mPoints, eLengths


def draw_polygon(point_list, fill=False, width=32, thickness=1):
    img = np.zeros((width, width))
    if fill:
        cv2.fillPoly(img, [point_list], color=255)
    else:
        cv2.polylines(img, [point_list], isClosed=True, color=255,
                      thickness=thickness, lineType=cv2.LINE_AA)
    return img


def draw_mask(img, fill, maskType, maskRate, r, point_list):
    if maskType == 'edge':
        if fill:
            pass
        else:
            mPoints, eLengths = get_mPoints_and_eLength(point_list)
            for p, mask_l in zip(mPoints, np.round(eLengths * maskRate).astype(int)):
                cv2.circle(img, p, mask_l, color=0, thickness=-1)
    elif maskType == 'vertex':
        mask_l = round(r * maskRate)
        for p in point_list:
            cv2.circle(img, p, mask_l, color=0, thickness=-1)


parser = argparse.ArgumentParser(description='Generate some polygons.')
parser.add_argument('-ds', '--dataset', type=str, required=True, default='train',
                    choices=['train', 'val', 'test', 'vis'],
                    help="the dataset you want to generate.")
parser.add_argument('-fn', '--figNum', type=int, default=2000,
                    help="the number of pictures to generate. Default: 2000.")
parser.add_argument('-f', '--fill', type=int, default=0,
                    choices=[0, 1],
                    help="whether or not to fill the polygons. Default: 0.")
parser.add_argument('-w', '--width', type=int, default=64,
                    help="the width of figure to generate.  Default: 64.")
parser.add_argument('-th', '--thickness', type=int, default=3,
                    help="thickness of polygon line. Default: 3.")
parser.add_argument('-an', '--angNums', type=str, default='3,4,5,6',
                    help="how many angles to be contained in one figure, so as the species of polygons.")
parser.add_argument('-fr', '--floatRates', type=str, default='0.05,0.05',
                    help="how much to float the center and angles in polygons.")
parser.add_argument('-mt', '--maskType', type=str, default='None',
                    choices=['vertex', 'edge', 'None'],
                    help="where to mask on polygons. Optional: vertex, edge or None(default)")
parser.add_argument('-mr', '--maskRate', type=float, default=0,
                    help="how much to mask on polygons. Default: 0.")
parser.add_argument('-s', '--seed', type=int, default=1026,
                    help="random seed for generating polygons. Default: 1026.")
args = parser.parse_args()

if __name__ == '__main__':
    fill = args.fill
    thickness = args.thickness
    dataset = args.dataset
    figNum = args.figNum
    width = args.width
    angNums = list(map(int, args.angNums.split(',')))
    float_rate = list(map(float, args.floatRates.split(',')))
    maskType = args.maskType
    maskRate = args.maskRate

    # control random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if fill:
        ROOT = f'Datasets/polygons_filled_{width}_{thickness}_{maskType}_{maskRate}/'
    else:
        ROOT = f'Datasets/polygons_unfilled_{width}_{thickness}_{maskType}_{maskRate}/'
    ROOT += dataset
    for angNum in tqdm(angNums):
        Path = os.path.join(ROOT, f'{angNum}')
        if not os.path.isdir(Path):
            os.makedirs(Path)
        for j in tqdm(range(figNum)):
            r, point_list = points(angNum, width, float_rate=float_rate)
            img = draw_polygon(point_list, fill, width, thickness)
            if maskType != 'None':
                draw_mask(img, fill, maskType, maskRate, r, point_list)
            img_address = os.path.join(Path, f"{angNum}_{j}.png")
            cv2.imwrite(img_address, img)
