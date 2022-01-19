# -*- coding: utf-8 -*-
# @Date    : 2022/1/8 13:25
# @Author  : WangYihao
# @File    : Generate_polygon.py

import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    fill = True
    dataset = 'train'
    picNum = 10000
    angNums = range(2, 7)
    width = 32
    if fill:
        ROOT = 'Datasets/polygons_filled_32/'
    else:
        ROOT = 'Datasets/polygons_unfilled_32/'
    ROOT += dataset
    for angNum in angNums:
        Path = os.path.join(ROOT, f'{angNum}')
        if not os.path.isdir(Path):
            os.makedirs(Path)
        for j in range(picNum):
            random.seed()
            point_list = points(n=angNum, width=width, r=13)
            img = draw_polygon(point_list, fill)
            img_address = os.path.join(Path, f"{angNum}_{j}.png")
            cv2.imwrite(img_address, img)

    # point_list = points(n=4, width=width, r=13)
    # img = draw_polygon(point_list)
    # plt.imshow(img, 'gray')
    # plt.show()
