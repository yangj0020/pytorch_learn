# @Time: 2021/9/23 12:00
# @Author: yangj0020
# @File: 2-2数据操作.py
# @Description:

import torch
import numpy as np

"""
2.2.1创建Tensor
"""
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())
print(x.shape)


"""
2.2.2操作
"""
x = torch.tensor(np.array(range(0, 15)).reshape(5, 3))
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)  # y = x + y
print(y)

print(x)
y = x[0, :]
y += 1
print(y)
print(x)

y = x.view(15)
z = x.view(-1, 5)
print(x.shape, y.shape, z.shape)
print(x)
x += 1
print(x)
print(y)

x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

x = torch.randn(1)
print(x)
print(x.item())


"""
2.2.3 广播机制
"""
x = torch.arange(1, 4).view(1, 3)
print(x)
y = torch.arange(1, 3).view(2, 1)
print(y)
print(x + y)


"""
2.2.4 运算的内存开销
"""
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
# y = y + x  # 开辟新的内存
y[:] = y + x  # 不开辟新的内存
torch.add(x, y, out=y)  # 不开辟新的内存
y.add_(x)  # 不开辟新的内存
y += x  # 不开辟新的内存
print(id(y) == id_before)


"""
2.2.5 Tensor和Numpy相互转换
"""
# Tensor转Numpy，共享内存
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

# Numpy转Tensor
a = np.ones(5)
b = torch.from_numpy(a)  # 共享内存
# b = torch.tensor(a)  # 不共享内存
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)