# @Time: 2021/10/13 14:44
# @Author: yangj0020
# @File: 9-3目标检测和边界框.py
# @Description:
import torch
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('../data/img/catdog.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

"""
9.3.1 边界框
"""
# bbox是bounding box的缩写
dog_bbox, cat_bbox = [80, 0, 300, 298], [350, 106, 455, 295]


def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()

