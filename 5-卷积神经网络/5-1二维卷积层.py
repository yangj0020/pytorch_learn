# @Time: 2021/10/1 13:29
# @Author: yangj0020
# @File: 5-1二维卷积层.py
# @Description:

import torch
from torch import nn

"""
5.1.1 二维互相关运算
"""
def corr2d(X, K):
    """
    互相关运算
    :param X: 输入数组
    :param K: 核数组
    :return: 数组 Y
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))


"""
5.1.2 二维卷积层
"""
class Conv2D(nn.Module):
    """
    自定义二位卷积层
    二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出
    卷积层的模型参数包括了卷积核和标量偏差
    在训练模型时，通常先对卷积核随机初始化，然后不断迭代卷积核和偏差
    """
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


"""
5.1.3 图像中物体边缘检测
卷积层的简单应用：检测图像中物体的边缘，即找到像素变化的位置
"""
X = torch.ones(6, 8)
X[:, 2: 6] = 0

K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)  # 边缘为1或-1，其它全为0.    卷积层可通过重复使用卷积核有效地表征局部空间


"""
5.1.4 通过数据学习核数组
"""
# 构造一个核数组形状是（1，2）的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l))

print('weight: ', conv2d.weight.data)
print('bias: ', conv2d.bias.data)
