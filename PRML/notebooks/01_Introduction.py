# introduction

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from prml.features import PolynomialFeatures
from prml.linear import (
    LinearRegressor,
    RidgeRegressor,
    BayesianRegressor
)

np.random.seed(1234)

# example: polynomial curve fitting

def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def func(x):
    return np.sin(2 * np.pi * x)

x_train, y_train = create_toy_data(func, 10, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = func(x_test)

plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.show()

# 先把现有情况画出来， 下面再拟合，并且拟合的曲线从0次， 1次到3次， 9次。并得出并不是拟合的阶数越高拟合越好，高的容易过拟合。反而3阶拟合更好

for i, degree in enumerate([0, 1, 3, 9]):
    plt.subplot(2, 2, i + 1)
    feature = PolynomialFeatures(degree)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)

    model = LinearRegressor()
    model.fit(X_train, y_train)
    y = model.predict(X_test)

    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="fitting")
    plt.ylim(-1.5, 1.5)
    plt.annotate("M={}".format(degree), xy=(-0.15, 1))
plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
plt.show()

# RMSE 均方根误差

"""
MSE: Mean Squared Error 
均方误差是指参数估计值与参数真值之差平方的期望值; 
MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。

MSE= 1/N ∑t=1N (observedt−predictedt)2

RMSE 
均方误差:均方根误差是均方误差的算术平方根

RMSE=1/N ∑t=1N (observedt−predictedt)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾⎷(根号包括整体)

MAE :Mean Absolute Error 
平均绝对误差是绝对误差的平均值 
平均绝对误差能更好地反映预测值误差的实际情况.

MAE=1/N ∑i=1N ∣(fi−yi)∣
fi表示预测值,yi表示真实值;

SD :standard Deviation 
标准差:标准差是方差的算术平方根。标准差能反映一个数据集的离散程度。平均数相同的两组组数据，标准差未必相同。

SD=1/N ∑i=1N (xi−u)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾⎷（根号包括整体）

u表示平均值(u=1/N(x1+.....xN))
"""

# 想要看看误差的情况，所以才引用了rmse。而且在这里才看到， 随着degree的增大， 训练误差减小， 测试误差增加
def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))

training_errors = []
test_errors = []

for i in range(10):
    feature = PolynomialFeatures(i)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)

    model = LinearRegressor()
    model.fit(X_train, y_train)
    y = model.predict(X_test)
    training_errors.append(rmse(model.predict(X_train), y_train))
    test_errors.append(rmse(model.predict(X_test), y_test + np.random.normal(scale=0.25, size=len(y_test))))

plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()

# regularization
feature = PolynomialFeatures(9)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model = RidgeRegressor(alpha=1e-3)
model.fit(X_train, y_train)
y = model.predict(X_test)

y = model.predict(X_test)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.annotate("M=9", xy=(-0.15, 1))
plt.show()

# bayesian curve fitting
# 只有贝叶斯，或者相关的和概率有关的预测，才会有y_std， 因为它的预测是一个概率分布， 不是一条线， 是一个范围， ，这个范围基本能把所有数据都包括在内， 这样看的话， 拟合效果还挺好
# 均值是一条拟合曲线， 方差给出了一个范围

model = BayesianRegressor(alpha=2e-3, beta=2)
model.fit(X_train, y_train)

y, y_err = model.predict(X_test, return_std=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="mean")
plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
plt.annotate("M=9", xy=(0.8, 1))
plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)
plt.show()
