# @Time: 2021/9/27 9:29
# @Author: yangj0020
# @File: 3-11模型选择-欠拟合和过拟合.py
# @Description:

import torch
import numpy as np
import sys
sys.path.append('..')
import  d2lzh_pytorch as d2l


"""
3.11.4 多项式函数拟合实验
"""

"""
3.11.4.1 生成数据集
"""
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


"""
3.11.4.2 定义、训练和测试模型
"""
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    """
    作图函数，其中y轴使用了对数尺度
    :param x_vals:
    :param y_vals:
    :param x_label:
    :param y_label:
    :param x2_vals:
    :param y2_vals:
    :param legend:
    :param figsize:
    :return:
    """
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以此处无需手动初始化

    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)


"""
3.11.4.3 三阶多项式函数拟合（正常）
"""
fit_and_plot(poly_features[: n_train, :], poly_features[n_train:, :], labels[: n_train], labels[n_train:])


"""
3.11.4.4 线性函数拟合（欠拟合）
"""
fit_and_plot(features[: n_train, :], features[n_train:, :], labels[: n_train], labels[n_train:])


"""
3.11.4.5 训练样本不足（过拟合）
"""
fit_and_plot(poly_features[0: 2, :], poly_features[n_train:, :], labels[0: 2], labels[n_train:])
