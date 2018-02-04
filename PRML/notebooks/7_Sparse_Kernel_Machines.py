# 如何生成待分类数据， 如何生成待回归数据

# sparse kernel machines
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

from prml.kernel import (
	RBF, 
	PolynomialKernel, 
	SupportVectorClassifier, 
	RelevanceVectorRegressor, 
	RelevanceVectorClassifier
)

np.random.seed(1234)

# 实验线性可分模型，所以只给出了三个点， 并且提供的是多项式核（degree = 1， 线性函数）
# maximum margin classifiers # 最大边界分类
x_train = np.array([
	[0., 2.], 
	[2., 0.], 
	[-1., -1.]])
y_train = np.array([1., 1., -1.])

model = SupportVectorClassifier(PolynomialKernel(degree = 1))
model.fit(x_train, y_train)
x0, x1 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
x = np.array([x0, x1]).reshape(2, -1).T
plt.scatter(x_train[:, 0], x_train[:, 1], s = 40, c = y_train, marker = "x")
plt.scatter(model.X[:, 0], model.X[:, 1], s = 100, facecolor = "none", edgecolor = "g")
cp = plt.contour(x0, x1, model.distance(x).reshape(100, 100), np.array([-1, 0, 1]), colors = "k", linestyles = ("dashed", "solid", "dashed")) # 等高线
plt.clabel(cp, fmt = 'y = %.f', inline = True, fontsize = 15) # 在等高线上加上标注
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.gca().set_aspect("equal", adjustable = "box")

# 线性不可分，但是界限是明显的， 需要用一条曲线区分，所以要引入RBF核函数
def create_toy_data():
	x = np.random.uniform(-1, 1, 100).reshape(-1, 2)
	y = x < 0
    y = (y[:, 0] * y[:, 1]).astype(np.float)
    return x, 1 - 2 * y

x_train, y_train = create_toy_data()

model = SupportVectorClassifier(RBF(np.ones(3)))
model.fit(x_train, y_train)

x0, x1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
x = np.array([x0, x1]).reshape(2, -1).T
plt.scatter(x_train[:, 0], x_train[:, 1], s=40, c=y_train, marker="x")
plt.scatter(model.X[:, 0], model.X[:, 1], s=100, facecolor="none", edgecolor="g")
plt.contour(
    x0, x1, model.distance(x).reshape(100, 100),
    np.arange(-1, 2), colors="k", linestyles=("dashed", "solid", "dashed"))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect("equal", adjustable="box")

# 有重叠的类分布，需要用曲线区分，所以要有RBF核函数，而且每一类都有等高线包起来
# overlapping class distributions
def create_toy_data():
	x0 = np.random.normal(size = 100).reshape(-1, 2) - 1.
	x1 = np.random.normal(size = 100).reshape(-1, 2) + 1.
	x = np.concatenate([x0, x1])
	y = np.concatenate([-np.ones(50), np.ones(50)]).astype(np.int)
	return x, y

x_train, y_train = create_toy_data()

model = SupportVectorClassifier(RBF(np.array([1. ,0.5, 0.5])), C = 1.)
model.fit(x_train, y_train)

x0, x1 = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
x = np.array([x0, x1]).reshape(2., -1).T
plt.scatter(x_train[:, 0], x_train[:, 1], s = 40, c = y_train, marker = "x")
plt.scatter(model.X[:, 0], model.X[:, 1], s = 100, facecolor = "none", edgecolor = "g")
plt.contour(x0, x1, model.distance(x).reshape(100, 100), np.arange(-1, 2), colors = "k", linestyles = ("dashed", "solid", "dashed"))
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect("equal", adjustable = "box")

# 以上都是SVM分类， 并且可以学习如何生成各中类分布的数据
##################################################################
# 回归拟合的数据比较好生成，就用一个函数，再加上一点服从正态分布的噪音即可

# Relevance vector machines 相关向量机

# RVM for regression

def create_toy_data(n = 10):
	x = np.linspace(0, 1, n)
	t = np.sin(2 * np.pi * x) + np.random.normal(scale = 0.1, size = n)
	return x, t

x_train, y_train = create_toy_data(n = 10)
x = np.linspace(0, 1, 100)

model = RelevanceVectorRegressor(RBF(np.array([1., 20.])))
model.fit(x_train, y_train)

y, y_std = model.predict(x)

plt.scatter(x_train, y_train, facecolor = "none", edgecolor = "g", label = "training")
plt.scatter(model.X.ravel(), model.t, s = 100, facecolor = "none", edgecolor = "b", label = "relevance vector")
plt.plot(x, y, color = "r", label = "predict mean")
plt.fill_between(x, y - y_std, y + y_std, color = "pink", alpha = 0.2, label = "predict std.")
plt.legend(loc = "best")
plt.show()

# RVM for classification
def create_toy_data():
	x0 = np.random.normal(size = 100).reshape(-1, 2) - 1.
	x1 = np.random.normal(size = 100).reshape(-1, 2) + 1.
	x = np.concatenate([x0, x1])
	y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int)
	return x, y

x_train, y_train = create_toy_data()

model = RelevanceVectorClassifier(RBF(np.array([1., 0.5, 0.5])))
model.fit(x_train, y_train)

x0, x1 = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
x = np.array([x0, x1]).reshape(2., -1).T
plt.scatter(x_train[:, 0], x_train[:, 1], s = 40, c = y_train, marker = "x")
plt.scatter(model.X[:, 0], model.X[:, 1], s = 100, facecolor = "none", edgecolor = "g")
plt.contour(x0, x1, model.predict_proba(x).reshape(100, 100), np.linspace(0, 1, 5), alpha = 0.2)
plt.colorbar()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect("equal", adjustable = "box")
