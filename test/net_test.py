import numpy as np
import torch
import os
import re
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
# @Time : 2021/2/24 上午9:19 
# @Author : glider 
# @File : net_model.py

"""
基于pytorch构建各种网络，
掌握不同网络的改进细节;
"""


def net_components():
    input=torch.randn(1,1,4,4)
    maxpool=nn.MaxPool2d(2,1)
    out=maxpool(input)
    print(out.shape)




# 线性拟合
def test1():
    x1 = torch.linspace(-100, 100, 1000)
    x2 = x1.view(1, 1000)
    x = torch.unsqueeze(x1, dim=1)
    print(x.shape)
    # 生成-100到100的1000个数的等差数列
    y = 2 * x2 + 1
    # 定义y=2x+1函数
    matrix = torch.nn.Linear(1000, 1000)
    # 定义一个1x1的矩阵
    optimizer = torch.optim.Adam(matrix.parameters(), lr=0.1)
    # 使用优化器求导更新矩阵权重
    for _ in range(2):
        # 训练100次
        value = matrix(x2)
        # value是x与矩阵相乘后的值
        score = torch.mean((value - y) ** 2)
        # 目标偏差，值为(value-y)的平方取均值，越接近0说明结果越准确
        matrix.zero_grad()
        score.backward()
        optimizer.step()
        # 根据求导结果更新权值
        print("第{}次训练权值结果:{}，结果偏差：{}".format(_, matrix.weight.data.numpy(), score))
    # 输出结果：
    # 第0次训练权值结果:[[0.9555]]，结果偏差：4377.27294921875
    # ...
    # 第99次训练权值结果:[[2.0048]]，结果偏差：0.10316929966211319

# BP 研究生期间搞了两年的东西就是这几句代码 搞毛线啊
def test2():
    # 所有数据多次迭代 每次计算loss 更新一次 而yolo是按照一个batch训练一次 循环次数为epoch 所有数据只用一次  err!!!
    w = 2
    b = 1
    noise = torch.rand(100, 1)
    x1 = torch.linspace(-1, 1, 100)
    x = torch.unsqueeze(x1, dim=1)  # a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
    print(x1)
    print(x.shape)
    # 因为输入层格式要为(-1, 1)，所以这里将(100)的格式转成(100, 1)
    y = w * x + b + noise
    # 拟合分布在y=2x+1上并且带有噪声的散点
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )
    # 自定义的网络，带有2个全连接层和一个tanh层
    loss_fun = torch.nn.MSELoss()
    # 定义损失函数为均方差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用adam作为优化器更新网络模型的权重，学习率为0.01

    plt.ion()
    # 图形交互
    for _ in range(1000):
        ax = plt.axes()
        output = model(x)
        # 数据向后传播（经过网络层的一次计算）
        loss = loss_fun(output, y)
        # 计算损失值
        # print("before zero_grad:{}".format(list(model.children())[0].weight.grad))
        # print("-"*100)
        model.zero_grad()
        # 优化器清空梯度
        # print("before zero_grad:{}".format(list(model.children())[0].weight.grad))
        # print("-"*100)
        # 通过注释地方可以对比发现执行zero_grad方法以后倒数梯度将会被清0
        # 如果不清空梯度的话，则会不断累加梯度，从而影响到当前梯度的计算
        loss.backward()
        # 向后传播，计算当前梯度，如果这步不执行，那么优化器更新时则会找不到梯度
        optimizer.step()
        # 优化器更新梯度参数，如果这步不执行，那么因为梯度没有发生改变，loss会一直计算最开始的那个梯度
        if _ % 20 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
            # print("w:", list(model.children())[0].weight.t() @ list(model.children())[-1].weight.t())
            # 通过这句可以查看权值变化，可以发现最后收敛到2附近

    plt.ioff()
    plt.show()


# cnn
def test3():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_feature(
                x))  # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = F.softmax(x)

            return x

        def num_flat_feature(self, x):  # x=x.view(x.size(0),-1)
            size = x.size()[1:]
            num_feature = 1
            for i in size:
                num_feature *= i
            return num_feature

    net = Net()
    print(net)
    input = Variable(torch.rand(1, 1, 32, 32))
    print(input)
    output1 = net.forward(input)  # 为啥不是调用内部函数forward
    # output=net(input) #这个地方就神奇了，明明没有定义__call__()函数啊，所以只能猜测是父类实现了，并且里面还调用了forward函数
    print(output1)
    # optimizer=optim.SGD(net.parameters(),lr=0.01)
    # optimizer.zero_grad()
    # loss= nn.CrossEntropyLoss(output1,target)
    # loss.backward()
    # optimizer.step()

# lecnn
def test4():
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            layer1 = nn.Sequential()
            layer1.add_module('conv1', nn.Conv2d(1, 6, 5))
            layer1.add_module('pool1', nn.MaxPool2d(2, 2))
            self.layer1 = layer1

            layer2 = nn.Sequential()
            layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
            layer2.add_module('pool2', nn.MaxPool2d(2, 2))
            self.layer2 = layer2

            layer3 = nn.Sequential()
            layer3.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
            layer3.add_module('fc2', nn.Linear(120, 84))
            layer3.add_module('fc3', nn.Linear(84, 10))
            self.layer3 = layer3

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.view(x.size(0), -1)  # 转换（降低）数据维度，进入全连接层
            x = self.layer3(x)
            return x

    # 代入数据检验
    y = torch.randn(1, 1, 32, 32)
    model = LeNet()
    model(y)
    print(model)


# alexnet
def test5():
    class AlexNet(nn.Module):
        def __init__(self, num_classes=1000):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
            return x

    net = AlexNet()
    print(net)


# vggnet
def test6():
    class VGG(nn.Module):
        def __init__(self, num_classes=100):
            super(VGG, self).__init__()
            layers = []
            in_dim = 3
            out_dim = 64
            for i in range(13):
                layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
                in_dim = out_dim
                if i == 1 or i == 3 or i == 6 or i == 9 or i == 12:
                    layers += [nn.MaxPool2d(2, 2)]
                    if i != 9:
                        out_dim *= 2
            self.feature = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )

        def forward(self, x):
            x = self.feature(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    vgg = VGG(21)
    input = torch.randn([1, 3, 224, 224])
    output = vgg(input)
    print(output)
    print(vgg)


# goolenet
def test7():
    class BasicConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding=0):
            super(BasicConv2d, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return F.relu(x, inplace=True)

    class Inceptionv2(nn.Module):
        def __init__(self):
            super(Inceptionv2, self).__init__()
            self.branch1 = BasicConv2d(192, 96, 1, 0)
            self.branch2 = nn.Sequential(BasicConv2d(192, 48, 1, 0), BasicConv2d(48, 64, 3, 1))
            self.branch3 = nn.Sequential(BasicConv2d(192, 64, 1, 0), BasicConv2d(64, 96, 3, 1),
                                         BasicConv2d(96, 96, 3, 1))
            self.branch4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,count_include_pad=False), BasicConv2d(192, 64, 1, 0))

        def forword(self, x):
            x0 = self.branch1(x)
            x1=self.branch2(x)
            x2=self.branch3(x)
            x3=self.branch4(x)
            out=torch.cat((x0,x1,x2,x3),1)
            return out

    input=torch.randn(1,192,32,32)
    net_inceptionv2=Inceptionv2()
    out=net_inceptionv2.forword(input)
    print(out.shape)


# resnet  残差块 三件套堆叠+输入
def test8():
    class Bottleneck(nn.Module):
        def __init__(self,in_dim,out_dim,stride=1):
            super(Bottleneck, self).__init__()
            self.bottleneck=nn.Sequential(
                nn.Conv2d(in_dim,in_dim,1,bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim,in_dim,3,1,bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim,out_dim,1,bias=False),
                nn.BatchNorm2d(out_dim),
            )
            self.relu=nn.ReLU(inplace=True)
            self.downsample=nn.Sequential(
                nn.Conv2d(in_dim,out_dim,1,1),
                nn.BatchNorm2d(out_dim),
            )

        def forword(self,x):
            identity=x
            out=self.bottleneck(x)
            identity=self.downsample(x)
            out+=identity
            out=self.relu(out)
            return out

#FPN
def test9():
    pass

if __name__ == "__main__":
    # test7()
    net_components()
