#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import random
import copy
import matplotlib.pyplot as plt


# SVM对象,用于存储相应的值
class SVMStruct:

    def __init__(self, x, y, C, Toler, kernelType):
        self.x = x  # 数据样本,每一行代表一个样本
        self.y = y  # 分类标签
        self.b = 0
        self.C = C  # C是离群点的权重，C越大表明离群点对目标函数影响越大，也就是越不希望看到离群点
        self.Toler = Toler  # 松弛变量
        self.numSamples = x.shape[0]    # 样本个数
        self.alphas = mat(zeros((self.numSamples, 1)))  # 所有样本的拉格朗日因子
        self.errorCache = mat(zeros((self.numSamples, 2)))  # 每次迭代的误差
        # 核函数类型
        # kernelType=("rbf", 1) rbf 表示使用径向基RBF函数作为核函数,其第二个参数不可为0
        # kernelType=("linear", 0) rbf 表示使用径向基RBF函数作为核函数,不会使用到第二个参数
        self.kernelType = kernelType
        # 计算x矩阵的核函数矩阵
        self.kernelMat = calcKernelMatrix(x, kernelType)


def calcKernelMatrix(x, kernelType):
    # 计算x矩阵的核函数矩阵
    n = x.shape[0]
    kernelMat = mat(zeros((n, n)))
    for i in range(n):
        kernelMat[:, i] = calcKernelValue(x, x[i, :], kernelType, n)
    return kernelMat


def calcKernelValue(x, xi, kernelType, n):
    # 根据选择的情况计算核函数矩阵
    kernelValue = mat(zeros((n, 1)))
    if kernelType[0] == "linear":
        kernelValue = x * xi.T
    elif kernelType[0] == "rbf":
        for i in range(n):
            diff = x[i, :] - xi
            kernelValue[i] = exp(diff * diff.T / (-2.0 * kernelType[1]**2))
    else:
        # 通过raise显示地引发异常。一旦执行了raise语句，raise后面的语句将不能执行
        raise NameError('请选择核函数的类型')
    return kernelValue


def SVMtrain(x, y, C, Toler, IterNum, kernelType):
    model = SVMStruct(x, y, C, Toler, kernelType)
    # 记录迭代结束后实际迭代次数
    iterCount = 0
    # 判断α的值是否发生了变化
    ischange = 1
    # 开始迭代,终止条件为：
    # 1、完成所有迭代
    # 2、α的值不再发生变化并且所有α(样本)符合KKT条件
    while(iterCount < IterNum and ischange > 0):
        ischange = 0
        for i in range(model.numSamples):
            ischange += innerLoop(model, i)
        iterCount += 1
    print(iterCount)

    return model


def innerLoop(model, i):
    Ei = calcError(model, i)
    # 判断该点是否符合KKT条件，符合就返回0.不符合进行更新
    if(FitKKT(model, i) == 0):
        return 0
    # 根据αi选择αj
    j, Ej = selectAlpha(model, i, Ei)
    # Python中的赋值操作（包括对象作为参数、返回值）不会开辟新的内存空间，它只是复制了新对象的引用
    # 浅拷贝会创建新对象，其内容是原对象的引用。浅拷贝有三种形式：切片操作，工厂函数，copy模块中的copy函数
    # 深拷贝拷贝了对象的所有元素，包括多层嵌套的元素
    alpha_i_old = copy.deepcopy(model.alphas[i])
    alpha_j_old = copy.deepcopy(model.alphas[j])
    # 计算边界L和H
    # if yi!=yj L=max(0,αj-αi) H=min(C,C+αj-αi)
    # if yi==yj L=max(0,αj+αi-C) H=min(C,αj+αi)
    if(model.y[i] != model.y[j]):
        L = max(0, model.alphas[j] - model.alphas[i])
        H = min(model.C, model.C + model.alphas[j] - model.alphas[i])
    else:
        L = max(0, model.alphas[j] + model.alphas[i] - model.C)
        H = min(model.C, model.alphas[j] + model.alphas[i])
    if L == H:
        return 0
    # 计算样本i和j之间的相似性
    ETA = 2.0 * model.kernelMat[i, j] - \
        model.kernelMat[i, i] - model.kernelMat[j, j]
    if ETA >= 0:
        return 0
    # 更新αj
    model.alphas[j] -= model.y[j] * (Ei - Ej) / ETA
    # αj必须在边界内，因此在计算出新的αj后要对其进行新的裁剪
    # if αj>H then αj=H
    # if L<=αj<=H then αj=αj
    # if αj>H then αj=H
    if(model.alphas[j] > H):
        model.alphas[j] = H
    if(model.alphas[j] < L):
        model.alphas[j] = L
    # 如果本次更新几乎没有变化就返回
    if abs(alpha_j_old - model.alphas[j]) < 0.00001:
        return 0
    # 更新αi  αi=αi+yi*yj*(aj_old-aj)
    model.alphas[i] += model.y[i] * model.y[j] * \
        (alpha_j_old - model.alphas[j])
    # 更新阀值b
    # b1=b-Ei-yi(αi-αi_old)<xi,xi>-yj(αj-αj_old)<xi,xj>
    # b2=b-Ej-yi(αi-αi_old)<xi,xj>-yj(αj-αj_old)<xj,xj>
    b1 = model.b - Ei - model.y[i] * (model.alphas[i] - alpha_i_old) * model.kernelMat[
        i, i] - model.y[j] * (model.alphas[j] - alpha_j_old) * model.kernelMat[i, j]
    b2 = model.b - Ej - model.y[i] * (model.alphas[i] - alpha_i_old) * model.kernelMat[
        i, j] - model.y[j] * (model.alphas[j] - alpha_j_old) * model.kernelMat[j, j]
    # if 0<αi<C then b=b1
    # if 0<αj<C then b=b2
    # if other then b=(b1+b2)/2
    if (0 < model.alphas[i]) and (model.alphas[i] < model.C):
        model.b = b1
    elif (0 < model.alphas[j]) and (model.alphas[j] < model.C):
        model.b = b2
    else:
        model.b = (b1 + b2) / 2.0
    updateError(model, i)
    updateError(model, j)
    return 1


def updateError(model, j):
    E = calcError(model, j)
    model.errorCache[j] = [1, E]


def selectAlpha(model, i, Ei):
    # [1, Ei] 1表示已经被优化
    model.errorCache[i] = [1, Ei]
    # nonzero()将布尔数组转换成整数数组
    # nonzero(x)返回x集合中为True的元素下标集合
    # 用于找出所有符合KKT条件的乘子的E
    Alphalist = nonzero(model.errorCache[:, 0].A)[0]
    maxStep = 0
    j = 0
    Ej = 0
    # 选择误差步长最大的最为αj
    if(len(Alphalist) > 1):
        # k 不能等于i和j
        for k in Alphalist:
            if k == i:
                continue
            Ek = calcError(model, k)
            if abs(Ek - Ei) > maxStep:
                maxStep = abs(Ek - Ei)
                j = k
                Ej = Ek
    # 如果是第一次 随机选择αj
    else:
        j = i
        while j == i:
            j = random.randint(0, model.numSamples - 1)
        Ej = calcError(model, j)
    return j, Ej


def calcError(model, i):
    f = multiply(model.alphas, model.y).T * model.kernelMat[:, i] + model.b
    return (f - model.y[i])


def FitKKT(model, i):
    E = calcError(model, i)
    # 约束条件1：0<=α<=C
    # 约束条件2：必须满足KKT条件
    #     1-1、if yf>=1 then α==0
    #     1-2、if yf<=1 then α==C
    #     1-3、if yf==1 then 0<α<C
    #     因此可以得到不满足KKT的条件
    #     2-1、if yf>=1 then α>0
    #     2-2、if yf<=1 then α<C
    #     2-3、if yf==1 then α==0 or α==C
    #     仔细考虑2-1，当yf=1 then α>0,符合1-3
    #     仔细考虑2-1，当yf=1 then α<C,符合1-3
    #     仔细考虑2-3，符合1-1和1-2
    #     因此得到：
    #     3-1、if yf>1 then α>0
    #     3-2、if yf<1 then α<C
    #     预测值与真实值之差 E=f-y
    #     yE=yf-yy because y∈(-1,1) so yE=fy-1
    #     4-1、if yE>0 then α>0
    #     4-2、if yE<0 then α<C
    #     我们在这里加入一个松弛变量Toler：
    #     1、if yE>Toler then α>0
    #     2、if yE<-Toler then α<C
    if((model.y[i] * E < -Toler and model.alphas[i] < model.C)
       or (model.y[i] * E > Toler and model.alphas[i] > 0)):
        return 1
    else:
        return 0


def SVMPredict(model, x, y):
    n = x.shape[0]
    # 加入核函数之后，新的fx=yi*ai*k<x,xi>+b
    # 又因为alphas大多数都为0，因此值用计算不为0的就可以
    supportVectorsIndex = nonzero(
        (model.alphas.A > 0) * (model.alphas.A < model.C))[0]
    supportVectors = model.x[supportVectorsIndex]
    supportVectorLabels = model.y[supportVectorsIndex]
    supportVectorAlphas = model.alphas[supportVectorsIndex]
    matchCount = 0
    for i in range(n):
        kernelValue = calcKernelValue(
            supportVectors, x[i, :], model.kernelType, n)
        predict = kernelValue.T * \
            multiply(supportVectorLabels, supportVectorAlphas) + model.b
        if sign(predict) == sign(y[i]):
            matchCount += 1
    accuracy = float(matchCount) / n
    return accuracy


def Draw(model):
    # 画出所有的点
    for i in range(model.numSamples):
        if model.y[i] == -1:
            plt.plot(model.x[i, 0], model.x[i, 1], 'or')
        elif model.y[i] == 1:
            plt.plot(model.x[i, 0], model.x[i, 1], 'ob')
    # 画出支持向量
    supportVectorsIndex = nonzero(
        (model.alphas.A > 0) * (model.alphas.A < model.C))[0]
    for i in supportVectorsIndex:
        plt.plot(model.x[i, 0], model.x[i, 1], 'oy')

    # 画出分类线
    w = zeros((2, 1))
    # 求wi=yi*ai*xi i=0,1,2...n
    for i in supportVectorsIndex:
        w += multiply(model.alphas[i] * model.y[i], model.x[i, :].T)
    min_x = min(model.x[:, 0])[0, 0]
    max_x = max(model.x[:, 0])[0, 0]
    y_min_x = float(-model.b - w[0] * min_x) / w[1]
    y_max_x = float(-model.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.show()
# ----------------------------------------开始------------------------------------
# 处理数据
x = []
y = []
C = 0.6
IterNum = 500
Toler = 0.001
files = open("/home/hadoop/Python/SVM/data.txt", "r")
for line in files.readlines():
    val = line.strip().split('\t')
    # windows 下 val = line.strip().split()
    # x.append([float(val[0]), float(val[1])])
    # y.append(float(val[2]))
    
    temp = val[0].split()     //very important
    x.append([float(temp[0]), float(temp[1])])
    y.append(float(temp[2]))
    
x = mat(x)
y = mat(y).T
# 训练模型
model = SVMtrain(x, y, C, Toler, IterNum, kernelType=("linear", 1))
for i in model.alphas:
    if(i > 0):
        print(i)
# 测试数据
predict = SVMPredict(model, x, y)
print(predict)
# 数据展示
Draw(model)
