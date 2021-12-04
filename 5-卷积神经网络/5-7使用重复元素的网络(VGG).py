# @Time: 2021/10/5 15:58
# @Author: yangj0020
# @File: 5-7使用重复元素的网络(VGG).py
# @Description:

import time
import torch
from torch import nn, optim

import sys
sys.path.append('..')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
5.7.1 VGG块
VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为3×33×3的卷积层后接上一个步幅为2、窗口形状为2×22×2的最大池化层。
卷积层保持输入的高和宽不变，而池化层则对其减半。
"""
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半
    return nn.Sequential(*blk)  # *blk表示，列表元素作为多个元素传入


"""
5.7.2 VGG网络
与AlexNet和LeNet一样，VGG网络由卷积层模块后接全连接层模块构成。
卷积层模块串联数个vgg_block，其超参数由变量conv_arch定义。该变量指定了每个VGG块里卷积层个数和输入输出通道数。
全连接模块则跟AlexNet中的一样。

构造一个VGG网络: 它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。
第一块的输入输出通道分别是1（因为下面要使用的Fashion-MNIST数据的通道数为1）和64，
之后每次对输出通道数翻倍，直到变为512。因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11。
"""
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次，变成了 224/32 = 7
fc_features = 512 * 7 * 7  # c * h * w
fc_hiddens_units = 4096  # 任意

# 实现 VGG-11
def vgg(conv_arch, fc_features, fc_hiddens_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module('vgg_block_' + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module('fc', nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(fc_features, fc_hiddens_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hiddens_units, fc_hiddens_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hiddens_units, 10)
    ))
    return net


# # 下面构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
# net = vgg(conv_arch, fc_features, fc_hiddens_units)
# X = torch.rand(1, 1, 224, 224)
#
# # named_children获取一级子模块及其名字（named_modules会返回所有子模块，包括子模块的子模块）
# for name, blk in net.named_children():
#     X = blk(X)
#     print(name, 'output shape: ', X.shape)


"""
5.7.3 获取数据和训练模型
因为VGG-11计算上比AlexNet更复杂，故此处测试时构造一个通道数更小或者说更窄的网络在Fashion-MNIST数据集上训练
"""
ratio = 8
small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                   (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hiddens_units // ratio)
# print(net)

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
