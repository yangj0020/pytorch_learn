# @Time: 2021/9/23 16:24
# @Author: yangj0020
# @File: 3-2线性回归的从0开始实现.py
# @Description:

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

"""
3.2.1生成数据集
"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # (1000 * 2)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # (1000, 1)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()


"""
读取数据
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 1000
    indices = list(range(num_examples))  # [0, 1000)
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break


"""
3.2.3 初始化模型参数
"""
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(True)
b.requires_grad_(True)


"""
3.2.4 定义模型
"""
def linreg(X, w, b):
    return torch.mm(X, w) + b


"""
3.2.5 定义损失函数
"""
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


"""
3.2.6 定义优化算法
"""
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


"""
3.2.7 定义模型
"""
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

print(len([w, b]))
print([w, b])

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
