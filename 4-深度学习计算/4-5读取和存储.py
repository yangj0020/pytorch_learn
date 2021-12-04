# @Time: 2021/9/30 14:59
# @Author: yangj0020
# @File: 4-5读取和存储.py
# @Description:

import torch
from torch import nn

"""
4.5.1 读取Tensor
"""
x = torch.ones(3)
# torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
# torch.save([x, y], 'xy.pt')

xy_list = torch.load('xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


"""
4.5.2 读取模型
"""
"""
4.5.2.1 state_dict
"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())


optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())


"""
4.5.2.2 保存和加载模型
"""
"""
4.5.2.2.1 保存和加载state_dict(推荐方式)
"""
X = torch.randn(2, 3)
Y = net(X)

PATH = './net.pt'
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)

"""
4.5.2.2.2 保存和加载整个模型
"""
torch.save(Y, './whole_net.pt')
Y3 = torch.load('./whole_net.pt')
print(Y == Y3)