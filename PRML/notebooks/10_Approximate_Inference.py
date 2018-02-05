"""
1.变分高斯混合模型
2.变分线性回归
3.变分逻辑斯特回归
"""




# 所谓的变分，简单地说，就是：用简单的分布q去近似复杂的分布p
# (也就是，对于没有战术配合的军队，军队的战斗力 ≈≈ 单兵的战斗力， 平均情况，
#	用单兵的战斗力近似军队的战斗力就是变分，就是平均场)

# 采用平均场就是将这种复杂的多元积分变成简单的多个一元积分，

"""
简单易懂的理解变分其实就是一句话：用简单的分布q去近似复杂的分布p。

首先，为什么要选择用变分推断？因为，大多数情况下后验分布很难求啊。
如果后验概率好求解的话我们直接EM就搞出来了。

当后验分布难于求解的时候我们就希望选择一些简单的分布来近似这些复杂的后验分布，
至于这种简单的分布怎么选，有很多方法比如：Bethe自由能，平均场定理。
而应用最广泛的要数平均场定理。

为什么？因为它假设各个变量之间相互独立砍断了所有变量之间的依赖关系。
这又有什么好处呢？我们拿一个不太恰当的例子来形象的说明一下：
用古代十字军东征来作为例子说明一下mean field。十字军组成以骑兵为主步兵为辅，
战之前骑兵手持重标枪首先冲击敌阵步兵手持刀斧跟随，一旦接战就成了单对单的决斗。

那么在每个人的战斗力基本相似的情况下某个人的战斗力可以由其他人的均值代替这是
平均场的思想。

这样在整个军队没有什么战术配合的情况下军队的战斗力可以由这些单兵
的战斗力来近似这是变分的思想。

(也就是，对于没有战术配合的军队，军队的战斗力 ≈≈ 单兵的战斗力， 平均情况，
	用单兵的战斗力近似军队的战斗力就是变分，就是平均场)

当求解Inference问题的时候相当于积分掉无关变量求边际分布，

如果变量维度过高，
积分就会变得非常困难，而且你积分的分布p又可能非常复杂因此就彻底将这条路堵死了。


采用平均场就是将这种复杂的多元积分变成简单的多个一元积分，
而且我们选择的q是指数族内的分布，更易于积分求解。

如果变量间的依赖关系很强怎么办？那就是structured mean field解决的问题了。
说到这里我们就知道了为什么要用变分，那么怎么用？过程很简单，推导很复杂。

整个过程只需要：

1、根据图模型写出联合分布
2、写出mean filed 的形式（给出变分参数及其生成隐变量的分布）
3、写出ELBO（为什么是ELBO？优化它跟优化KL divergence等价，
KL divergence因为含有后验分布不好优化）
4、求偏导进行变分参数学习这样就搞定了！要点都有了，
具体怎么推怎么理解还得多看亲自推一遍。
"""

"""
选择 A Tutorial on Variational Bayesian Inference 来举例。
首先要建立intuition, 应该选择简化的变分推断问题：
即Variational Bayes 和Mean-Field假设。
其次，熟悉经常会遇到的几个概念：Approximate Inference(近似推断), Energy（能量）, Entropy（熵）, Proxy（代理）
这段话值得反复咀嚼：
Variational Bayes is a particular variational method which aims to 
find some approximate joint distribution Q(x; θ) over hidden variables
 x to approximate the true joint P(x), and defines ‘closeness’ as the
  KL divergence KL[Q(x; θ)||P(x)]. 

变分贝叶斯是一种专门的变分方法， 主要用于找到一些近似的联合分布Q(x； θ)(包含隐变量x)来近似真正的概率P（x）。并用KL散度来定义近似程度。

  简言之：就是选择合适的分布（函数）来逼近真实的后验概率分布（对于多个隐藏因子
  的情况，就是联合分布）。

  因为是概率分布，所以相似度或者距离就用KL值来表达。


  也因此，可以把这理解成一个泛函问题。所以它是一种Approximate Inference的方法。
  之所以approximate, 是因为这时Exact Inference计算复杂度太高（求后验概率的
  贝叶斯公式中分母的问题）。而Approximate的时候可以从特定的distribution 
  family中选取q, 来方便计算。上面可以作为宏观的理解。深入的理解需要耐心看懂推
  导公式，比较长这里就不贴了
"""

"""

Variational Bayesian methods are primarily used for two purposes:
1. To provide an analytical approximation to the posterior probability 
of the unobserved variables, in order to do statistical inference over 
these variables.
为不可观测变量的后验概率提供分析性近似， 为了对这些变量做统计推断。

2.To derive a lower bound for the marginal likelihood (sometimes called 
	the "evidence") of the observed data (i.e. the marginal probability 
	of the data given the model, with marginalization performed over
	unobserved variables). 
为了产生观测数据的边际似然的下界 （有时候也叫证据evidence）
（例如， 要求输入模型的数据的边际概率， 就要对未观测变量执行边际化操作，类似于积分掉无关变量）

This is typically used for performing model 
selection, the general idea being that a higher marginal likelihood 
for a given model indicates a better fit of the data by that model 
and hence a greater probability that the model in question was the 
one that generated the data. (See also the Bayes factor article.
这个的典型应用是模型选择， 大概思想就是：一个模型的边际似然越高，说明对输入模型的数据的拟合程度越好。
因此概率越高说明越有可能是这个模型产生的这些数据

前面两位答主说的主要是第1点， 不过在深度学习中第2点更常见。

来看深度学习中两类强大的概率模型：基于隐变量和基于配分函数的模型。
1）
2）

它们都需要算积分，而说到快速估算积分，非常自然的选择是 Importance weighted sampling，即
3）

现在问题变成了如何选择一个q(x; θ)，使得估算的效率最高。不难看出，f(x)和q(x)越接近，估算就越稳定。
另一方面，我们知道，E[log f] <= logE[f]，等号成立当且仅当f为常数。因此，假如我们关心的是log ∫f(x) dx，那么我们得到的
	就是一个下界。对它作argmax就可以得到最佳的q(x; θ).
"""

# approximate inference
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

from prml.rv import VariationalGaussianMixture
from prml.features import PolynomialFeatures
from prml.linear import (
	VariationalLinearRegressor, 
	VariationalLogisticRegressor
)

np.random.seed(1234)

# illustration: variational mixture of gaussians
x1 = np.random.normal(size = (100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size = (100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size = (100, 2))
x3 += np.array([0, 5])
x_train = np.vstack((x1, x2, x3))

x0, x1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
x = np.array([x0, x1]).reshape(2, -1).T

vgmm = VariationalGaussianMixture(n_components = 6)
vgmm.fit(x_train)

plt.scatter(x_train[:, 0], x_train[:, 1], c = vgmm.classify(x_train))
plt.contour(x0, x1, vgmm.pdf(x).reshape(100, 100)) # 等高线
plt.xlim(-10, 10, 100)
plt.ylim(-10, 10, 100)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()


vgmm = VariationalGaussianMixture(n_components = 6)
vgmm._init_params(x_train)
params = np.hstack([param.flatten() for param in vgmm.get_params()])
fig = plt.figure()
colors = np.array(["r", "orange", "y", "g", "b", "purple"])
frames = []
for _ in range(100):
	plt.xlim(-10, 10)
	plt.ylim(-10, 10)
	plt.gca().set_aspect('equal', adjustable = 'box')
	r = vgmm._variational_expectation(x_train)
	imgs = [plt.scatter(x_train[:, 0], x_train[:, 1], c = colors[np.argmax(r, -1)])]
	for i in range(vgmm.n_components):
		if vgmm.component_size[i] > 1:
			imgs.append(plt.scatter(vgmm.mu[i, 0], vgmm.mu[i, 1], 100, colors[i], "X", lw = 2, edgecolors = "white"))
	frames.append(imgs)
	vgmm._variational_maximization(x_train, r)
	new_params = np.hstack([param.flatten() for param in vgmm.get_params()])
	if np.allclose(new_params, params):
		break
	else:
		params = np.copy(new_params)
plt.close()
plt.rcParams['animation.html'] = 'html5'
anim = animation.ArtistAnimation(fig, frames)
anim

# variational linear regression
def create_toy_data(func, sample_size, std, domain = [0, 1]):
	x = np.linspace(domain[0], domain[1], sample_size)
	np.random.shuffle(x)
	t = func(x) + np.random.normal(scale = std, size = x.shape)
	return x, t 

def cubic(x):
	return x * (x - 5) * (x + 5)

x_train, y_train = create_toy_data(cubic, 10, 10., [-5, 5])
x = np.linspace(-5, 5, 100)
y = cubic(x)

feature = PolynomialFeatures(degree = 3)
X_train = feature.transform(x_train)
X = feature.transform(x)

vlr = VariationalLinearRegressor(beta = 0.01)
vlr.fit(X_train, y_train)
y_mean, y_std = vlr.predict(X, return_std = True)
plt.scatter(x_train, y_train, s = 100, facecolor = "none", edgecolor = "b")
plt.plot(x, y, c = "g", label = "$\sin(2\pi x)$")
plt.plot(x, y_mean, c = "r", label = "prediction")
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha = 0.2, color = "pink")
plt.legend()
plt.show()

# variational logistic regression
def create_toy_data(add_outliers = False, add_class = False):
	x0 = np.random.normal(size = 50).reshape(-1, 2) - 3.
	x1 = np.random.normal(size = 50).reshape(-1, 2) + 3.
	return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)
x_train, y_train = create_toy_data()
x0, x1 = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
x = np.array([x0, x1]).reshape(2, -1).T
feature = PolynomialFeatures(degree = 1)
X_train = feature.transform(x_train)
X = feature.transform(x)

vlr = VariationalLogisticRegressor()
vlr.fit(X_train, y_train)
y = vlr.proba(X).reshape(100, 100)

plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train)
plt.contourf(x0, x1, y, np.array([0., 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.]), alpha = 0.2)
plt.colorbar()
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()
# 线性回归和逻辑斯特回归画图的时候不一样 不知道是两种回归做的事情，拟合的内容不同，还是故意画成不同的。

# 能力配得上心气儿， 态度得配得上欲望
