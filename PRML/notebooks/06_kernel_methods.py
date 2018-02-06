# kernel methods

import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

from prml.kernel import (
	PolynomialKernel,
	RBF,
	GaussianProcessClassifier,
	GaussianProcessRegressor
)

def create_toy_data(func, n = 10, std = 1., domain = [0., 1.]):
	x = np.linspace(domain[0], domain[1], n)
	t = func(x) + np.random.normal(scale = std, size = n)
	return x, t

def sinusoidal(x):
	return np.sin(2 * np.pi * x)

# dual representation 对偶表示 
# 许多线性参数模型都可以通过对偶表示的形式表达为核函数的形式，也就是说，哪里体现对偶了呢？有核函数呗

x_train, y_train = create_toy_data(sinusoidal, n = 10, std = 0.1)
x = np.linspace(0, 1, 100)

model = GaussianProcessRegressor(kernel = PolynomialKernel(3, 1.), beta = int(1e10))
model.fit(x_train, y_train)

y = model.predict(x)
plt.scatter(x_train, y_train, facecolor = "none", edgecolor = "b", color = "blue", label = "training")
plt.plot(x, sinusoidal(x), color = "g", label = "sin$(2\pi x)$")
plt.plot(x, y, color = "r", label = "gpr")
plt.show()

# 上面用的核函数是多项式核函数， 下面用的是RBF，效果与区别嘛， 还是等实践的时候再比较吧

# gaussian processes
# gaussian processes for regression

x_train, y_train = create_toy_data(sinusoidal, n=7, std=0.1, domain=[0., 0.7])
x = np.linspace(0, 1, 100)

model = GaussianProcessRegressor(kernel=RBF(np.array([1., 15.])), beta=100)
model.fit(x_train, y_train)

y, y_std = model.predict(x, with_error=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr")
plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
plt.show()

# learning the hyperparameters (就是有参数迭代， 这个迭代次数是在fit里面设定的， 你也会发现，迭代之后，拟合效果更好)

x_train, y_train = create_toy_data(sinusoidal, n=7, std=0.1, domain=[0., 0.7])
x = np.linspace(0, 1, 100)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1.])), beta=100)
model.fit(x_train, y_train)
y, y_std = model.predict(x, with_error=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr {}".format(model.kernel.params))
plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
plt.legend()

plt.subplot(1, 2, 2)
model.fit(x_train, y_train, iter_max=100)
y, y_std = model.predict(x, with_error=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr {}".format(np.round(model.kernel.params, 2)))
plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
plt.legend()
plt.show()


# automatic relevance determination 自相关决策 （是不是使用了高斯分类，就用了ARD？）

"""
说到ARD，不得不提RVM， 它是类似于SVM的一种分类方法，但是基于贝叶斯框架，要借助概率的思想思考问题。

RVM有着与 SVM类似的判别公式， 如 样本到分类面的距离f：
f(x, w) = ∑wn * K(x, xn) + w0 = W * Φ
只有参数w， 那么我们就假设w的先验服从高斯分布。借用某种方法可以移除不相关的点。增加稀疏性，只保留少数相关的点


在模式分类中，相关向量机使用基于贝叶斯框架的系数学习模型， 在先验参数的结构下，基于
自相关决策理论ARD来移除不相关的点， 从而获得稀疏化的模型， 具备着良好的稀疏性和泛化能力。

对于二值分类问题， 给定一定的训练样本集{xn, tn}, 输入向量xn ∈R^n, 样本标记tn ∈{1, -1}，样本到分类面的距离f为：
f(x, w) = ∑wn * K(x, xn) + w0 = W * Φ
其中w = (w0, w1, w2, ..., wn), Φ = (1, Φ1, ..., Φn)^T为基函数，  Φn = K(x, xn)

RVM核函数假设使用的是RBF核函数：
K(x, y) = exp(- ||x-y||^2 / σ2)
这里不要求基函数为正定的， 因此核函数不必要满足mercer条件。

相关向量机的分类模型就是计算输入属于期望类别的后验概率， 实际上， 相关向量机的分类模型将
f(x, w)通过logistic sigmoid函数:
λ(y) = 1 / (1 + e ^(-y))
转换线性模型， 则似然估计概率可以写为（伯努利分布）， 假设样本独立同分布情况下， 有：
p(t|w) = Π λ{f(xn; w)}^((1+tn)/2) * [1 - λ{f(xn; w)}]^((1-tn)/2)

为了保证模型的稀疏性， 相关向量机提出了通过在参数w上定义受超参数α控制的Gaussian先验概率，
在贝叶斯框架下进行机器学习，利用自相关决策理论（ARD）来移除不相关的点， 从而实现稀疏化。
即假设权重向量w的第i个分量wi服从均值为0， 方差为αi^-1的Gaussian分布。
定义权重的高斯先验：
p(w|α) = Π N(wi|0, αi^-1)

其中α为N+1维参数。 这样为每一个权值（或基函数）配置独立的超参数是稀疏贝叶斯模型最显著的特点，
也是导致模型具有稀疏性的根本原因。由于这种先验概率分布是一种自动相关判定ARD先验分布， 
模型训练结束后，许多权值为0， 与它对应的基函数将被删除，实现了模型的稀疏化。
而非零权值的基函数所对应的样本向量被称为相关向量， 这种学习机杯称为相关向量机。

https://max.book118.com/html/2015/0610/18747834.shtm
"""
def create_toy_data_3d(func, n = 10, std = 1.): # 看来生成三维数据挺简单的
	x0 = np.linspace(0, 1, n)
	x1 = x0 + np.random.normal(scale = std., size = n)
	x2 = np.random.normal(scale = std, size = n)
	t = func(x0) + np.random.normal(scale = std, size = n)
	return np.vstack((x0, x1, x2)).T, t

x_train, y_train = create_toy_data_3d(sinusoidal, n=20, std=0.1)
x0 = np.linspace(0, 1, 100)
x1 = x0 + np.random.normal(scale=0.1, size=100)
x2 = np.random.normal(scale=0.1, size=100)
x = np.vstack((x0, x1, x2)).T

model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1., 1., 1.])), beta=100)
model.fit(x_train, y_train)
y, y_std = model.predict(x, with_error=True)
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 0], y_train, facecolor="none", edgecolor="b", label="training")
plt.plot(x[:, 0], sinusoidal(x[:, 0]), color="g", label="$\sin(2\pi x)$")
plt.plot(x[:, 0], y, color="r", label="gpr {}".format(model.kernel.params))
plt.fill_between(x[:, 0], y - y_std, y + y_std, color="pink", alpha=0.5, label="gpr std.")
plt.legend()
plt.ylim(-1.5, 1.5)

model.fit(x_train, y_train, iter_max=100, learning_rate=0.001)
y, y_std = model.predict(x, with_error=True)
plt.subplot(1, 2, 2)
plt.scatter(x_train[:, 0], y_train, facecolor="none", edgecolor="b", label="training")
plt.plot(x[:, 0], sinusoidal(x[:, 0]), color="g", label="$\sin(2\pi x)$")
plt.plot(x[:, 0], y, color="r", label="gpr {}".format(np.round(model.kernel.params, 2)))
plt.fill_between(x[:, 0], y - y_std, y + y_std, color="pink", alpha=0.2, label="gpr std.")
plt.legend()
plt.ylim(-1.5, 1.5)
plt.show()

# 说实话， 我没看出来哪里体现ARD的？ 而且ARD到底怎么做到让一部分w变为0的？以后再慢慢琢磨
# 因为在做高斯模型拟合的时候会计算协方差矩阵
# Gaussian processes for classification

def create_toy_data():
    x0 = np.random.normal(size=50).reshape(-1, 2)
    x1 = np.random.normal(size=50).reshape(-1, 2) + 2.
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)[:, None]

x_train, y_train = create_toy_data()
x0, x1 = np.meshgrid(np.linspace(-4, 6, 100), np.linspace(-4, 6, 100))
x = np.array([x0, x1]).reshape(2, -1).T

model = GaussianProcessClassifier(RBF(np.array([1., 7., 7.])))
model.fit(x_train, y_train)
y = model.predict(x)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x0, x1, y.reshape(100, 100), levels=np.linspace(0,1,3), alpha=0.2)
plt.colorbar()
plt.xlim(-4, 6)
plt.ylim(-4, 6)
plt.gca().set_aspect('equal', adjustable='box')

# 这个分类效果怎么样呢？加上等高线之后， 感觉连混合的部分也能分出来呢？至于效果， 还是等到真正实践的时候再比较吧

