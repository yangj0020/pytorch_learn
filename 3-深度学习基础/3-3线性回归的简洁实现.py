# @Time: 2021/9/23 20:23
# @Author: yangj0020
# @File: 3-3线性回归的简洁实现.py
# @Description:

import numpy as np
import torch

"""
3.3.1 生成数据集
"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

"""
3.3.2 读取数据
"""
import torch.utils.data as Data

batch_size = 10

dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

"""
3.3.3 定义模型
"""
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, n_feature):  # n_feature表示特征数量
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
# print(net)
# print(net.linear)

# # 事实上我们还可以用nn.Sequential来更加方便地搭建网络，
# # Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中。
# # 写法一
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此处还可以传入其它层
# )
# # 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module......
# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, 1))
#     # ......
# ]))
# print(net)
# print(net[0])

# 查看模型的所有可学习参数
# for param in net.parameters():
#     print(param)


"""
3.3.4 初始化模型参数
"""
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

"""
3.3.5 定义损失函数
"""
loss = nn.MSELoss()

"""
3.3.6 定义优化算法
"""
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)

# 还可以为不同子网络设置不同的学习率，这在finetune时经常用到。例：
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)

# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1  # 学习率为之前的0.1倍

"""
3.3.7 训练模型
"""
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)
