# @Time: 2021/10/1 14:53
# @Author: yangj0020
# @File: 5-3多输入通道和多输出通道.py
# @Description:

import torch
from torch import nn
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

"""
5.3.1 多输入通道
"""
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
print(corr2d_multi_in(X, K))


"""
5.3.2 多输出通道
"""
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关运算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])  # 输出通道数 * 输入通道数 * 高 * 宽
print(corr2d_multi_in_out(X, K))  # 第一个通道的结果与之前输入数组X与多输入通道、单输出通道核的计算结果一致


"""
5.3.3 1 x 1卷积层
"""
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)
