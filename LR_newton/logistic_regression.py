#!/usr/bin/python
# -*- coding: utf-8 -*-

# matplotlib.pyplot 用于绘制2D图表
import matplotlib.pyplot as plt
from numpy import *


class LR:
    # 数据的行数
    __row = 0
    # 数据的列数
    __col = 0
    # 迭代次数
    __numIterations = 10
    # 训练集合
    __trainData = []
    # θ参数
    __theta = []
    # Y值
    __Y = []
    # 损失
    __Cost = []

    def __init__(self, data):
        self.__row, self.__col = shape(data)
        self.__trainData = data[:, 0:self.__col - 1]
        self.__Y = data[:, self.__col - 1:self.__col]
        self.__col = self.__col - 1
        # 系数的初始值为0  如果是其他值有可能会算不出结果
        self.__theta = mat(zeros((self.__col, 1)))

    # 设置迭代次数
    def setnumIterations(self, numIterations):
        self.__numIterations = numIterations

    # 获取theta
    def getTheta(self):
        return self.__theta

    # 获取cost损失
    def getCost(self):
        return self.__Cost

    # 训练数据模型
    def train(self):
        # 存储损失
        self.__Cost = mat(zeros((self.__numIterations, 1)))
        for i in range(0, self.__numIterations):
            # 更新Theta
            self.__updateTheta(i)

    # 迭代theta
    def __updateTheta(self, i):
        # 获取预测值h(x) 1.0 / (1 + exp(-z))
        h = 1.0 / (1 + exp(-(self.__trainData * self.__theta)))
        # 获取损失
        self.__getCost(i, h)
        # .T 矩阵的转置
        # 一阶导数矩阵算法 (1 / m) * x.T * (h-y)
        J = multiply(1.0 / self.__row,
                     self.__trainData.T * (h - self.__Y))
        # 获取Hession矩阵
        # getA() 矩阵转换为数组
        # diag(x) 生成对角线为x其余为0的矩阵
        # Hession矩阵算法(1 / m) * x.T * U * x   U表示用(h * (1 - h))构成对角，其余为0的矩阵
        H = multiply(1.0 / self.__row, self.__trainData.T *
                     diag(multiply(h, (1 - h)).T.getA()
                          [0]) * self.__trainData)
        # .I 矩阵的逆
        self.__theta = self.__theta - H.I * J

    # 计算损失
    def __getCost(self, i, h):
        l1 = self.__Y.T * log(h)
        l2 = (1 - self.__Y).T * log((1 - h))
        self.__Cost[i, :] = multiply(1.0 / self.__row, sum(-l1 - l2))

    # 画图
    def draw(self):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(0, self.__row):
            if(self.__Y[i] == 1):
                x1.append(self.__trainData[i, 1])
                y1.append(self.__trainData[i, 2])
            else:
                x2.append(self.__trainData[i, 1])
                y2.append(self.__trainData[i, 2])
        # plt.figure()创建一个绘图对象
        # 第一个参数表示绘图对象的序号，当序号相同时候不会创建新的对象，而是指向该序号的对象。
        # 第二个参数表示绘图对象的尺寸。参数值 * 80
        plt.figure(0, figsize=(8, 5))
        fig = plt.figure(0)
        # 创建一个画布,参数含义(m,n,k）将画布分为m * n块，这个图像在第k块
        ax = fig.add_subplot(1, 1, 1)
        # scatter散列图 s 数据点的大小 c颜色 marker形状
        ax.scatter(x1, y1, s=30, c="red", marker="o")
        ax.scatter(x2, y2, s=30, c="green")
        x = arange(20, 80, 10)
        a = float(self.__theta[1])
        b = float(self.__theta[0])
        c = float(self.__theta[2])
        y = (a * x + b) / (-c)
        # plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
        # label : 曲线名字 添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式
        # color : 指定曲线的颜色        linewidth : 指定曲线的宽度
        # plt.plot(x,z,"b--",label="$cos(x^2)$") 第三个参数"b--"指定曲线的颜色和线型(蓝色 虚线)
        ax.plot(x, y)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(["hx", "x1y1", "x2y2"])
        plt.show()

# -------------------------开始-----------------------
x = open("/home/hadoop/Python/LR/ex4x.dat", "r")
y = open("/home/hadoop/Python/LR/ex4y.dat", "r")
data = []
for i in x:
    temp = [1]
    val = i.split()
    for j in val:
        temp.append(float(j))
    temp.append(float(y.readline()))
    data.append(temp)

lr = LR(mat(data))
lr.setnumIterations(7)
lr.train()
print(lr.getTheta())
print(lr.getCost())
lr.draw()
