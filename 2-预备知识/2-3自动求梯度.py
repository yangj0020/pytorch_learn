# @Time: 2021/9/23 15:09
# @Author: yangj0020
# @File: 2-3自动求梯度.py
# @Description:

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.arange(3, 7).view(2, 2)
print(a, a * 3, a - 1)
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

print(x)
out.backward()
print(x.grad)

out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

print('====================')
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)

v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)  # r = torch.sum(z * v)，然后求r对x的导数
print(x.grad)

print('=======================')
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

y3.backward()
print(x.grad)

print('==================')
x = torch.ones(1, requires_grad=True)

print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100

y.backward()
print(x)
print(x.grad)
