# @Time: 2021/9/24 16:32
# @Author: yangj0020
# @File: 3-8多层感知机.py
# @Description:


import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import d2lzh_pytorch as d21


def xyplot(x_vals, y_vals, name):
    d21.set_figsize(figsize=(5, 2.5))
    d21.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d21.plt.xlabel('x')
    d21.plt.ylabel(name + '(x)')
    plt.show()


"""
3.8.2.1 relu函数
"""
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')

y.sum().backward()
xyplot(x, x.grad, 'grad of relu')


"""
3.8.2.2 sigmoid函数
"""
y = x.sigmoid()
xyplot(x, y, 'sigmoid')

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')


"""
3.8.2.3 tanh函数
"""
y = x.tanh()
xyplot(x, y, 'tanh')

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
