#!/usr/bin/python3
# -*- coding: utf-8 -*-
from numpy import *
import operator


# 这些数据的背后意义：某个妹纸想要找一个男朋友，她收集了婚恋网上面对她感兴趣男士的一些数据，
# 第一个是一年的飞行里程，第二个是每年玩游戏的时间(10小时为单位），第三个是每年吃的冰淇淋量。
# 第四个是结果：3表示很感兴趣，2表示感兴趣，1表示没兴趣

class Classify:

    def __init__(self, trainData, trainlables):
        self.size = trainData.shape[0]
        self.trainData = self.normalization(trainData)
        self.trainlables = trainlables

    # 对数据进行归一化处理
    # 归一化分为：
    # 线性函数归一化(Min-Max scaling)：x = y / s   y = x - x.min  s = x.max - x.min
    # 0均值标准化(Z-score standardization)：z = (x - μ) / σ,μ表示均值σ表示方差
    # 0均值标准化要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。这里采用线性函数归一化
    def normalization(self, trainData):
        # data m*n    min与max用法相同
        # data.min() 获取data中的最小值，一个数字
        # data.min(0) data中每列最小的数据，1*n数组
        # data.min(1) data中每行最小的数据，1*m数组
        minary = trainData.min(0)
        maxary = trainData.max(0)
        interval = maxary - minary
        # tile(minary, (self.m, 1))表示构建一个self.m*1的矩阵。矩阵的每个元素是minary
        # a = [1, 2, 3]
        # tile(a, (2, 3))为[[1,2,3,1,2,3,1,2,3] [1,2,3,1,2,3,1,2,3]]
        Xnormal = trainData - tile(minary, (self.size, 1))
        Xnormal = trainData / tile(interval, (self.size, 1))
        return Xnormal

    # 预测分类
    def predict(self, testData, testlables, k):
        testData = self.normalization(testData)
        size = len(testData)
        errorCount = 0.0
        for i in range(size):
            result = self.classify(testData[i], k)
            if(result != testlables[i]):
                errorCount += 1.0
        return errorCount / size

    def classify(self, inMat, k):
        # 计算欧式距离
        diff = tile(inMat, (self.size, 1)) - self.trainData
        diffP = pow(diff.A, 2)
        distance = sqrt(diffP.sum(1))
        # argsort函数返回的是数组值从小到大的索引值
        index = distance.argsort()
        classCount = {}
        for i in range(k):
            l = self.trainlables[index[i]]
            # get(l, 0) 查找字典中key=l的value,如果没有该属性返回0
            classCount[l] = classCount.get(l, 0) + 1
        # key=operator.itemgetter(1) 定义函数key，获取对象的第1个域的值
        sortedClassCount = sorted(
            classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


# 获取数据
def readFile(filename):
    files = open(filename, "r", encoding="utf-8")
    trainData = []
    lables = []
    for line in files.readlines():
        item = [float(i) for i in line.strip().split()]
        trainData.append(item[:- 1])
        lables.append(item[- 1])
    files.close()
    return mat(trainData), lables
# -----------------------------------开始----------------------------
# 用trainData作为训练，同时用trainData作为测试
trainData, trainlables = readFile("/home/hadoop/Python/KNN/train.txt")
model = Classify(trainData, trainlables)
errorRate = model.predict(trainData, trainlables, 20)
print("the error rate is :%f" % (errorRate))
