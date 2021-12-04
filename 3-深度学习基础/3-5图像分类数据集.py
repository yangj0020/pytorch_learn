# @Time: 2021/9/24 9:36
# @Author: yangj0020
# @File: 3-5图像分类数据集.py
# @Description:

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
import d2lzh_pytorch as d21


"""
3.5.1 获取数据集
"""
mnist_train = torchvision.datasets.FashionMNIST(root='D:/college/labC306/deeplearning/pytorch_learn/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='D:/college/labC306/deeplearning/pytorch_learn/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height X Width


def get_fashion_mnist_labels(labels):
    """
    将数值标签转换为相应的文本标签
    :param labels:
    :return:
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """
    在一行画出多张图像和对应标签
    :param images:
    :param labels:
    :return:
    """
    d21.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


"""
3.5.2 读取小批量
"""
batch_size = 256
if sys.platform.startswith('win'):
    print(sys.platform)
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

start = time.time()
i = 0
for X, y in train_iter:
    if i == 1:
        break
    print('-------------------')
    print(X, y)
    print(X.shape, y.shape)
    i += 1
    continue
# print('%.2f sec' % (time.time() - start))
