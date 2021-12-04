# @Time: 2021/10/11 15:23
# @Author: yangj0020
# @File: 7-2梯度下降和随机梯度下降.py
# @Description:

import numpy as np
import torch
import math
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l


def gd(eta):
    x = 10
    results = [10]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10, x', x)
    return results


res = gd(0.2)


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x * x for x in f_line])
    d2l.plt.plot(res, [x * x for x in res], '-o')
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()


show_trace(res)

"""
7.2.2 学习率
"""
show_trace(gd(0.05))
show_trace(gd(1.1))

"""
7.2.3 多维梯度下降
"""


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    d2l.plt.show()


eta = 0.1


def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)


show_trace_2d(f_2d, train_2d(gd_2d))

"""
7.2.4 随机梯度下降
"""
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

show_trace_2d(f_2d, train_2d(sgd_2d))
