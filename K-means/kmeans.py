#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import random
import matplotlib.pyplot as plt

plt.figure(0, figsize=(16, 6))
fig = plt.figure(0)


class Kmeans:

    def __init__(self, x, k, maxIterations):
        self.m, self.n = x.shape
        # 第一个参数表示其所属的类簇
        self.x = x
        self.k = k
        # 实际迭代次数
        self.count = 0
        if(self.k > self.m):
            raise NameError("设置的类簇个数大于数据样本数")
        self.maxIterations = maxIterations
        self.CenterPoint = zeros((k, self.n - 1))
        self.setKCenterPoint()

    def train(self):
        # 是否有点的类簇发生变化
        ischange = 1
        while(self.count < self.maxIterations and ischange > 0):
            ischange = 0
            for i in range(self.m):
                ischange += self.computeClass(i)
            if(ischange > 0):
                self.changeCenterPoint()
            self.count += 1

    def computeClass(self, i):
        k = -1
        temp = -1
        for j in range(self.k):
            flag = sqrt(
                sum(pow((self.x[i, 1:] - self.CenterPoint[j]).A, 2)))
            if(flag < temp or temp == -1):
                temp = flag
                k = j
        if(k == self.x[i, 0]):
            return 0
        else:
            self.x[i, 0] = k
            return 1

    # 重新计算类簇中心点
    def changeCenterPoint(self):
        self.CenterPoint = zeros((self.k, self.n - 1))
        for i in range(self.k):
            temp = nonzero(self.x[:, 0] == i)[0]
            for j in temp:
                self.CenterPoint[i] += self.x[j, 1:].A[0]
            self.CenterPoint[i] /= len(temp)
            
    # https://www.cnblogs.com/kemaswill/archive/2013/01/26/2877434.html

    # 计算所有数据点到其最近的中心点的平均距离cost
    # 一般来说,同样的迭代次数和算法跑的次数,这个值越小代表聚类的效果越好
    # 但是在实际情况下,我们还要考虑到聚类结果的可解释性,不能一味的选择使computeCost结果值最小的那个K
    def computeCost(self):
        cost = 0
        for i in range(self.m):
            cost += sqrt(sum(pow((self.x[i, 1:] -
                                  self.CenterPoint[int(self.x[i, 0])]).A, 2)))
        return cost / self.m

    
    '''
    确定K个初始类簇中心点
  首先随机选择一个点作为第一个初始类簇中心点，然后选择距离该点最远的那个点作为第二个初始类簇中心点，然后再选择距离前两个点的最近距离最大的点作
  为第三个初始类簇的中心点，以此类推，直至选出K个初始类簇中心点。
    '''
    # 选取K个类簇中心点的初始值
    # 1、随机选择一个点作为第一个类簇中心
    # 2、    1）选择距离已有类簇中心最近的点
    #        2）选择距离该点最远的点
    # 循环第2步直到选出k个类簇中心点
    def setKCenterPoint(self):
        # 随机选取一个作为第一个中心点
        i = random.randint(0, self.m - 1)
        self.CenterPoint[0] = self.x[i, 1:]
        for j in range(1, self.k):
            nearestPoint = self.nearestPoint(j)
            farthestPoint = self.farthestPoint(nearestPoint)
            self.CenterPoint[j] = farthestPoint

    # 选择最近的点
    def nearestPoint(self, i):
        # 获取第二个类簇中心点时，最近的点就是第一个中心点
        if(i == 1):
            return self.CenterPoint[0]
        else:
            nearestPoint = []
            temp = -1
            for j in range(self.m):
                flag = 0
                # i=2 表示获取第3个类簇中心点，获取距离前2个类簇中心点最近的点
                for m in range(i):
                    flag += sqrt(sum(pow((self.x[j, 1:] -
                                          self.CenterPoint[m]).A, 2)))
                if(flag < temp or temp == -1):
                    temp = flag
                    nearestPoint = self.x[j, 1:]
            return nearestPoint

    # 选择最远的点
    def farthestPoint(self, nearestPoint):
        farthestPoint = []
        temp = -1
        for i in range(self.m):
            flag = sqrt(sum(pow((self.x[i, 1:] - nearestPoint).A, 2)))
            if(flag > temp and self.x[i, 1:] not in self.CenterPoint):
                temp = flag
                farthestPoint = self.x[i, 1:]
        return farthestPoint

    def draw(self):
        ax = fig.add_subplot(1, 2, 2)
        mark1 = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        mark2 = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 根据聚类结果画出所有的点
        for i in range(self.m):
            ax.plot(self.x[i, 1], self.x[i, 2], mark1[int(self.x[i, 0])])
        # 所有类簇的中心点
        for j in range(self.k):
            ax.plot(self.CenterPoint[j, 0], self.CenterPoint[
                     j, 1], mark2[j], markersize=12)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

# ------------------------------开始-------------------------
x = []
files = open("/home/hadoop/Python/K-means/K-means.txt", "r")
for val in files.readlines():
    line = val.strip().split()
    data = [-1]
    for item in line:
        data.append(float(item))
    x.append(data)
x = mat(x)
# 通过computeCost计算cost确定K值
k = [1, 2, 3, 4, 5, 6]
cost = []
for i in k:
    model = Kmeans(x, i, 20)
    model.train()
    cost.append(model.computeCost())
ax = fig.add_subplot(1, 2, 1)
ax.plot(k, cost, "g")
plt.xlabel("k")
plt.ylabel("cost")

model = Kmeans(x, 4, 20)
model.train()
print(model.CenterPoint)
model.draw()
