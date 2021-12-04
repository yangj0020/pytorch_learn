# @Time: 2021/9/30 9:20
# @Author: yangj0020
# @File: 4-1模型构造.py
# @Description:

import torch
from torch import nn

"""
4.1.1 继承Module类来构造模型
"""
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数参数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))

"""
4.1.2 Module的子类
"""
"""
4.1.2.1 Sequential类
"""
from collections import OrderedDict


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net)
print(net(X))


"""
4.1.2.2 ModuleList类
"""
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net[-1])
print(net)
# print(net(torch.zeros(1, 784)))  # 会报NotImplementedError。
# ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），
# 而且没有实现forward功能需要自己实现，所以上面执行net(torch.zeros(1, 784))会报NotImplementedError
# Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。


# 加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中
class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])


class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]

net1 = Module_ModuleList()
net2 = Module_List()

print('net1:')
print(net1)
for p in net1.parameters():
    print(p.size())
    print(p)

print('net2:')
print(net2)
for p in net2.parameters():
    print(p.size())


"""
4.1.2.3 ModuleDict类
"""
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})
net['output'] = nn.Linear(256, 10)
print(net['linear'])
print(net.act)
print(net.output)
print(net)
# print(net(torch.zeros(1, 784))) # 会报NotImplementedError。（类似ModuleList）
# ModuleDict里的所有模块的参数会被自动添加到整个网络中


"""
4.1.3 构造复杂的模型
"""
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


X = torch.rand(2, 20)
net = FancyMLP()
print('4.1.3-------------------------------------')
print(net)
print(net(X))

# 因为FancyMLP和Sequential类都是Module类的子类，所以可以嵌套调用它们
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)
print(net)
print(net(X))