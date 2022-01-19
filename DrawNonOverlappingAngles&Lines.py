import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def angle(width=100, r=None, thickness=1):
    if r is None:
        r = int(np.floor(width / 4.))
    x1 = random.randint(0 + r, width - r)
    y1 = random.randint(0 + r, width - r)
    plot1 = (x1, y1)
    alpha = random.uniform(0, 2 * np.pi)
    theta = random.uniform(0.1, np.pi * 2 / 3)
    x2 = x1 + r * np.cos(alpha)
    y2 = y1 - r * np.sin(alpha)
    plot2 = (int(x2), int(y2))
    x3 = x1 + r * np.cos(alpha + theta)
    y3 = y1 - r * np.sin(alpha + theta)
    plot3 = (int(x3), int(y3))
    img = np.zeros((width, width), dtype=np.uint8)
    cv2.line(img, plot1, plot2, color=255, thickness=thickness)
    cv2.line(img, plot1, plot3, color=255, thickness=thickness)

    return img


def line(img, width=100):
    l = random.randint(0.2 * width, 1.4 * width)
    x1 = random.randint(0, width)
    y1 = random.randint(0, width)
    plot1 = (x1, y1)
    rl = 0.1 * width
    while rl < 0.2 * width:
        random.seed()
        alpha = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + l * np.cos(alpha))
        y2 = int(y1 - l * np.sin(alpha))
        x2 = max(0, x2)
        x2 = min(width, x2)
        y2 = max(0, y2)
        y2 = min(width, y2)
        rl = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    plot2 = (x2, y2)
    cv2.line(img, plot1, plot2, 255)
    return img


dataset = 'test'
picNum = 2000
angNums = range(1, 7)
lineNum = 0
width = 64
background = 'black'
# Path = './angle%d,%d' % (angNum, lineNum)
# Path = 'E:/D_L_/SimpleGeometry/geometry pictrues/angle&line 1,1'
data_type = f'angle_{background}_bg_{width}'
for angNum in angNums:
    Path = os.path.join('Datasets', f'{data_type}/{dataset}/{angNum}')
    if not os.path.isdir(Path):
        os.makedirs(Path)
    n_cols = int(np.ceil(np.sqrt(angNum + lineNum)))
    blankNum = n_cols ** 2 - (angNum + lineNum)
    data_file = open(f'Datasets/{data_type}/{dataset}/labels.txt', 'a')

    for j in range(picNum):
        imgList = []
        for i in range(angNum):
            img = angle(width)
            imgList.append(img)
        for i in range(lineNum):
            img = np.zeros((width, width), dtype=np.uint8)
            img = line(img)
            imgList.append(img)
        for i in range(blankNum):
            img = np.zeros((width, width), dtype=np.uint8)
            imgList.append(img)

        random.shuffle(imgList)
        img = np.zeros((n_cols * width, n_cols * width), dtype=np.uint8)
        count = 0
        for x in range(n_cols):
            for y in range(n_cols):
                img[x * width:(x + 1) * width, y * width:(y + 1) * width] = imgList[count]
                count += 1
        img_address = Path + f"/{j}.png"
        if background == 'white':
            img = 255 - img
        cv2.imwrite(img_address, img)
        data_file.write(f'{img_address} {angNum}\n')
