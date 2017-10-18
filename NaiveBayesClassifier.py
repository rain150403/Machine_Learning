#!/usr/bin/python3
# -*- coding: utf-8 -*-
from numpy import *
import random
import math
import operator

'''
朴素贝叶斯分类
理论参考：
http://blog.csdn.net/lsldd/article/details/41542107
程序参考：
http://python.jobbole.com/81019/
数据下载地址：
https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data
实现步骤：
1、数据预处理
    将数据转换为float数组
    将数据分割80%训练，20%测试
    将数据按照类别划分
2、计算所有分类对应特征的平均值和标准差
3、测试数据
    计算数据对应个分类的概率，概率大的就是该分类
    P(x) = (1 / (sqrt(2 * π) * σ)) * exp(-pow((x - η), 2) / (2 * pow(σ, 2)))
    η = 均值   σ = 标准差
    输出结果
'''


class NaiveBayesClassifier:

    def __init__(self, trainData):
        self.trainData = {}
        self.avg = {}
        self.stdev = {}
        self.Probability = []
        self.accuracy = 0.0
        self.formatData(trainData)
        self.getAvgByClass()
        self.getStDev()

    def formatData(self, data):
        '''按照规定格式化数据：{'类别(0,1,2,3)':[[数据],[],[]]}'''
        self.trainData = {}
        for item in data:
            if(item[-1] not in self.trainData):
                self.trainData[int(item[-1])] = []
            self.trainData[int(item[-1])].append(item[:-1])
        for key in self.trainData.keys():
            # 转换为numpy.array格式方便后续的计算
            self.trainData[key] = array(self.trainData[key])

    def getAvgByClass(self):
        '''每个类中每个属性的均值'''
        self.avg = {}
        for key in self.trainData.keys():
            self.avg[key] = self.trainData[key].mean(0)

    def getStDev(self):
        '''每个类中每个属性的标准差'''
        self.stdev = {}
        for key in self.trainData.keys():
            self.stdev[key] = self.trainData[key].std(0)

    def calculateProbability(self, item):
        '''计算概率，判断其在那个分类概率大就返回那个分类'''
        result = []
        for key in self.trainData.keys():
            temp = (1 / (sqrt(2 * pi) * self.stdev[key])) * exp(-pow((item[:-1] - self.avg[key]),
                                                                     2) / (2 * pow(self.stdev[key], 2)))
            result.append([key, log(temp).sum()])
        result = sorted(result, key=operator.itemgetter(1), reverse=True)
        return result[0][0]

    def predict(self, testData):
        '''测试数据'''
        self.Probability = []
        for item in testData:
            self.Probability.append(
                [item[-1], self.calculateProbability(item)])

    def getAccuracy(self):
        '''计算数据的正确率'''
        correct = 0.0
        size = len(self.Probability)
        for item in self.Probability:
            if(item[0] == item[1]):
                correct += 1.0
        self.accuracy = (correct / size) * 100.0


def readFile(filename):
    '''获取数据'''
    files = open(filename, "r", encoding="utf-8")
    trainData = []
    for line in files.readlines():
        item = [float(i) for i in line.strip().split(",")]
        trainData.append(item)
    files.close()
    return trainData


def splitCollection(collection, splitRatio):
    '''根据比例，随机分割数据'''
    size = int(len(collection) * splitRatio)
    ones = []
    others = []
    index = []
    while(len(index) < size):
        i = random.randint(0, len(collection) - 1)
        if(i not in index):
            index.append(i)
    for i in range(len(collection)):
        if(i in index):
            ones.append(collection[i])
        else:
            others.append(collection[i])
    return ones, others

'''-------------------------------------开始----------------------------------------'''
if __name__ == "__main__":
    data = readFile("/home/hadoop/Python/NaiveBayesClassifier/data")
    # trainData, testData = splitCollection(data, 0.68)
    model = NaiveBayesClassifier(data)
    model.predict(data)
    model.getAccuracy()
    # print(("Split {0} rows into train={1} and test={2} rows").format(
    #     len(data), len(trainData), len(testData)))
    print(('Accuracy: {0}%').format(model.accuracy))
