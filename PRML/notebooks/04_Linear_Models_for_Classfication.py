"""
	BayesianLogisticRegressor,
	LeastSquaresClassifier,
	LinearDiscriminantAnalyzer,
	LogisticRegressor, 也可以用于分类问题
	Perceptron,
	SoftmaxRegressor  用于多类问题分类  

	应该思考各类分类方法的优缺点，及效果比较，适用情况等
"""


# linear models for classification
import numpy as np 
import matplotlib.pyplot as plt 
# %matplotlib inline

from prml.features import PolynomialFeatures
from prml.linear import (
	BayesianLogisticRegressor,
	LeastSquaresClassifier,
	LinearDiscriminantAnalyzer,
	LogisticRegressor,
	Perceptron,
	SoftmaxRegressor
)

np.random.seed(1234)

# 看看是想加入一个异常值，还是想加入一个新的类
def create_toy_data(add_outliers = False, add_class = False):
	x0 = np.random.normal(size = 50).reshape(-1, 2) - 1
	x1 = np.random.normal(size = 50).reshape(-1, 2) + 1.
	if add_outliers:
		x_1 = np.random.normal(size = 10).reshape(-1, 2) + np.array([5., 10.])
		return np.concatenate([x0, x1, x_1]), np.concatenate([np.zeros(25), np.ones(30)]).astyp(np.int)
	if add_class:
		x2 = np.random.normal(size = 50).reshape(-1, 2) + 3.
		return np.concatenate([x0, x1, x2]), np.concatenate([np.zeros(25), np.ones(25), 2 + np.zeros(25)]).astype(np.int)
	return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)


# discriminant functions 判别函数
# least squares for classification
x_train, y_train = create_toy_data()
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model = LeastSquaresClassifier()
model.fit(X_train, y_train)
y = model.classify(X_test)

plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train)
plt.contourf(x1_test, x2_test, y.reshape(100, 100), alpha = 0.2, levels = np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()
# 最简单的不一定是最常见的，最小二乘法可以用于分类

x_train, y_train = create_toy_data(add_outliers = True)
x1_test, x2_test = np.meshgrid(np.linspace(-5, 15, 100), np.linspcae(-5, 15, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

least_squares = LeastSquaresClassifier()
least_squares.fit(X_train, y_train)
y_ls = least_squares.classify(X_test)

logistic_regressor = LogisticRegressor()
logistic_regressor.fit(X_train, y_train)
y_lr = logistic_regressor.classify(X_test)

plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_ls.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 15)
plt.ylim(-5, 15)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Least Squares")
plt.subplot(1, 2, 2)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_lr.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 15)
plt.ylim(-5, 15)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Logistic Regression")
plt.show()

"""
从图像上看， 好像是logistic regression效果更好一些， least squares是一根竖线，而logistic 是一根斜线，而两类的分界线最好的是斜线
"""
# 一共分了三种情况：正常的两类， 有异常值， 多加了一类，下面看最后一种情况

x_train, y_train = create_toy_data(add_class=True)
x1_test, x2_test = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

least_squares = LeastSquaresClassifier()
least_squares.fit(X_train, y_train)
y_ls = least_squares.classify(X_test)

logistic_regressor = SoftmaxRegressor()
logistic_regressor.fit(X_train, y_train, max_iter=1000, learning_rate=0.01)
y_lr = logistic_regressor.classify(X_test)

plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_ls.reshape(100, 100), alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]))
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Least squares")
plt.subplot(1, 2, 2)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_lr.reshape(100, 100), alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]))
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Softmax Regression")
plt.show()

# 也不知道这个问题有没有意义，但存疑， 两种分类方法的分类面是怎么产生的？

# Fisher’s linear discriminant Fisher的线性判别

x_train, y_train = create_toy_data()
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

model = LinearDiscriminantAnalyzer()
model.fit(x_train, y_train)
y = model.classify(x_test)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# 和第一种用最小二乘法解决单纯的两类分类问题的效果差不多，其实这样的数据集比不出什么

#以上是判别函数
#####################################################################
# 下面将讲概率判别模型

# probabilistic discriminative models
# logistic regression

x_train, y_train = create_toy_data()
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(degree=1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model = LogisticRegressor()
model.fit(X_train, y_train)
y = model.proba(X_test)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y.reshape(100, 100), np.linspace(0, 1, 5), alpha=0.2)
plt.colorbar()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# 与上无异， 难道logistics也算概率判别模型？ 既然如此，自然要生成概率， 那么就不调用classify函数，而是调用proba函数， 怪不得会看到有这两个函数了， 明白了
# 多类逻辑回归就是soft max

# multiclass logistic regression
x_train, y_train = create_toy_data(add_class=True)
x1, x2 = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
x = np.array([x1, x2]).reshape(2, -1).T

feature = PolynomialFeatures(1)
X_train = feature.transform(x_train)
X = feature.transform(x)

model = SoftmaxRegressor()
model.fit(X_train, y_train, max_iter=1000, learning_rate=0.01)
y = model.classify(X)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1, x2, y.reshape(100, 100), alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]))
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# bayesian logistic regression
x_train, y_train = create_toy_data()
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(degree=1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model = BayesianLogisticRegressor(alpha=1.)
model.fit(X_train, y_train, max_iter=1000)
y = model.proba(X_test)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y.reshape(100, 100), np.linspace(0, 1, 5), alpha=0.2)
plt.colorbar()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# 只是简单的两类分类， 在分界面处， 越来越细致， 像SVM似的
