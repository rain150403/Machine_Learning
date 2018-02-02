# probability distributions
# 提到随机变量概率分布，说简单也简单说难也难，就等到用到的时候再根据实例选择分布来学习吧

import numpy as np 
import matplotlib.pyplot as plt
# %matplotlib inline

from prml.rv import (
	Bernoulli,
	Beta,
	Categorical,
	Dirichlet,
	Gamma,
	Gaussian,
	MultivariateGaussian,
	MultivariateGaussianMixture,
	StudentsT,
	Uniform
)

np.random.seed(1234)

# binary variables
model = Bernoulli()
model.fit(np.array([0., 1., 1., 1.]))
print(model)

# the beta distributions
# a, b对beta分布的影响是什么？
x = np.linspace(0, 1, 100)
for i, [a, b] in enumerate([[0.1, 0.1], [1, 1], [2, 3], [8, 4]]):
	plt.subplot(2, 2, i + 1)
	beta = Beta(a, b)
	plt.xlim(0, 1)
	plt.ylim(0, 3)
	plt.plot(x, beta.pdf(x))
	plt.annotate("a = {}".format(a), (0.1, 2.5))
	plt.annotate("b = {}".format(b), (0.1, 2.1))
plt.show()

"""
beta分布是概率的概率分布，我不知道它击中的概率是多少，但是可以知道每一个击中概率的可能性大小。
也就是在没有看到他打球的时候，我们就对它的命中率有一个猜测，这个可以用beta分布来描述，
而且先验作为参数，beta分布会越来越精准。那么如何根据先验确定参数呢？
先验：平均击球率是0.27， 范围是0.21~0.35， 由此可以取α = 81， β = 219
原因：
1） beta分布的均值是α /（ α + β） = 81 / （81 + 219） = 0.27
2）从图中可以看到这个分布主要落在了（0.2， 0.35）间， 这是从经验中得出的合理范围
我们的x轴就表示各个击球率的取值，x对应的y值就是这个击球率所对应的概率。

Beta(α0+hits, β0 + misses）

对于一个我们不知道概率是什么，而又有一些合理的猜测时，beta分布能很好的作为一个表示概率的概率分布。




用一句话来说，beta分布可以看作一个概率的概率分布，
当你不知道一个东西的具体概率是多少时，它可以给出了所有概率出现的可能性大小。

举一个简单的例子，熟悉棒球运动的都知道有一个指标就是棒球击球率(batting average)，
就是用一个运动员击中的球数除以击球的总数，我们一般认为0.266是正常水平的击球率，
而如果击球率高达0.3就被认为是非常优秀的。
现在有一个棒球运动员，我们希望能够预测他在这一赛季中的棒球击球率是多少。
你可能就会直接计算棒球击球率，用击中的数除以击球数，但是如果这个棒球运动员只打了一次，
而且还命中了，那么他就击球率就是100%了，这显然是不合理的，因为根据棒球的历史信息，
我们知道这个击球率应该是0.215到0.36之间才对啊。
对于这个问题，我们可以用一个二项分布表示（一系列成功或失败），
一个最好的方法来表示这些经验（在统计中称为先验信息）就是用beta分布，
这表示在我们没有看到这个运动员打球之前，我们就有了一个大概的范围。
beta分布的定义域是(0,1)这就跟概率的范围是一样的。
接下来我们将这些先验信息转换为beta分布的参数，我们知道一个击球率应该是平均0.27左右，
而他的范围是0.21到0.35，那么根据这个信息，我们可以取α=81,β=219
之所以取这两个参数是因为：
1）beta分布的均值是
2）从图中可以看到这个分布主要落在了(0.2,0.35)间，这是从经验中得出的合理的范围。
在这个例子里，我们的x轴就表示各个击球率的取值，x对应的y值就是这个击球率所对应的概率。
也就是说beta分布可以看作一个概率的概率分布。

那么有了先验信息后，现在我们考虑一个运动员只打一次球，那么他现在的数据就是”1中;1击”。
这时候我们就可以更新我们的分布了，让这个曲线做一些移动去适应我们的新信息。
beta分布在数学上就给我们提供了这一性质，他与二项分布是共轭先验的（Conjugate_prior）。
所谓共轭先验就是先验分布是beta分布，而后验分布同样是beta分布。
结果很简单： Beta(α0+hits, β0 + misses）
其中α0和β0是一开始的参数，在这里是81和219。所以在这一例子里，α增加了1(击中了一次)。
β没有增加(没有漏球)。这就是我们的新的beta分布Beta(81+1,219)，我们跟原来的比较一下：
可以看到这个分布其实没多大变化，这是因为只打了1次球并不能说明什么问题。
但是如果我们得到了更多的数据，假设一共打了300次，其中击中了100次，200次没击中，
那么这一新分布就是： 
注意到这个曲线变得更加尖，并且平移到了一个右边的位置，表示比平均水平要高。
一个有趣的事情是，根据这个新的beta分布，我们可以得出他的数学期望为：，
这一结果要比直接的估计要小  。
你可能已经意识到，我们事实上就是在这个运动员在击球之前可以理解为他已经成功了81次，
失败了219次这样一个先验信息。
因此，对于一个我们不知道概率是什么，而又有一些合理的猜测时，beta分布能很好的作为一个表示概率的概率分布。
"""

beta = Beta(2, 2)
plt.subplot(2, 1, 1)
plt.xlim(0, 1)
plt.ylim(0, 2)
plt.plot(x, beta.pdf(x))
plt.annotate("prior", (0.1, 1.5))

model = Bernoulli(mu=beta)
model.fit(np.array([1]))
plt.subplot(2, 1, 2)
plt.xlim(0, 1)
plt.ylim(0, 2)
plt.plot(x, model.mu.pdf(x))
plt.annotate("posterior", (0.1, 1.5))

plt.show()

"""
现在假设我们有这样几类概率： p(θ)（先验分布）,p(θ|x)（后验分布）, p(X), p(X|θ) （似然函数）

它们之间的关系可以通过贝叶斯公式进行连接： 后验分布 = 似然函数* 先验分布/ P(X)

Beta is the conjugate prior of Binomial.
https://www.cnblogs.com/simayuhe/p/5143538.html
好像在这个链接里可以看见似然， 后验， 先验， 共轭等概念，对我理解有帮助


"""
print("Maximum likelihood estimation")
model = Bernoulli()
model.fit(np.array([1]))
print("{} out of 10000 is 1".format(model.draw(10000).sum()))

print("Bayesian estimation")
model = Bernoulli(mu=Beta(1, 1))
model.fit(np.array([1]))
print("{} out of 10000 is 1".format(model.draw(10000).sum()))


########################################################
#
#以上介绍的是二元变量， beta， Bernoulli
#
#下面看看多元变量：categorical，dirichlet
#
#########################################################


# multinomial variables

model = Categorical()
model.fit(np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]))
print(model)

# the dirichlet distribution
mu = Dirichlet(alpha=np.ones(3))
model = Categorical(mu=mu)
print(model)

model.fit(np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
print(model)


############################################################
#
# 下面是高斯分布:
# 1) N 在这里起的作用是什么？
# 2）高斯分布的极大似然估计
# 3）高斯分布的贝叶斯推断
############################################################
# the gaussian distribution

uniform = Uniform(low=0, high=1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.annotate("N=1", (0.1, 4.5))
plt.hist(uniform.draw(100000), bins=20, normed=True)

plt.subplot(1, 3, 2)
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.annotate("N=2", (0.1, 4.5))
plt.hist(0.5 * (uniform.draw(100000) + uniform.draw(100000)), bins=20, normed=True)

plt.subplot(1, 3, 3)
plt.xlim(0, 1)
plt.ylim(0, 5)
sample = 0
for _ in range(10):
    sample = sample + uniform.draw(100000)
plt.annotate("N=10", (0.1, 4.5))
plt.hist(sample * 0.1, bins=20, normed=True)

plt.show()

# maximum likelihood for the gaussian

X = np.random.normal(loc=1., scale=2., size=(100, 2))
gaussian = MultivariateGaussian()
gaussian.fit(X)
print(gaussian)

x, y = np.meshgrid(
    np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
p = gaussian.pdf(
    np.array([x, y]).reshape(2, -1).T).reshape(100, 100)
plt.scatter(X[:, 0], X[:, 1], facecolor="none", edgecolor="steelblue")
plt.contour(x, y, p)
plt.show()

# Bayesian inference for the Gaussian

mu = Gaussian(0, 0.1)
model = Gaussian(mu, 0.1)

x = np.linspace(-1, 1, 100)
plt.plot(x, model.mu.pdf(x), label="N=0")

model.fit(np.random.normal(loc=0.8, scale=0.1, size=1))
plt.plot(x, model.mu.pdf(x), label="N=1")

model.fit(np.random.normal(loc=0.8, scale=0.1, size=1))
plt.plot(x, model.mu.pdf(x), label="N=2")

model.fit(np.random.normal(loc=0.8, scale=0.1, size=8))
plt.plot(x, model.mu.pdf(x), label="N=10")

plt.xlim(-1, 1)
plt.ylim(0, 5)
plt.legend()
plt.show()

###################################################
# gamma分布
"""
alpha （一般为整数）代表一件事发生的次数；beta代表它发生一次的概率（或者叫速率）。
那么gamma 分布就代表这么一件事发生alpha 次所需要时间的分布。例如alpha=1 就是指数分布

Gamma分布即为多个独立且相同分布（iid）的指数分布变量的和的分布。
Gamma分布中的参数α称为形状参数（shape parameter），β称为尺度参数（scale parameter）

随机变量X为 等到第α件事发生所需之等候时间
EX = α/β， var（X）= α / β^2
"""

x = np.linspace(0, 2, 100)
for i, [a, b] in enumerate([[0.1, 0.1], [1, 1], [2, 3], [4, 6]]):
    plt.subplot(2, 2, i + 1)
    gamma = Gamma(a, b)
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.plot(x, gamma.pdf(x))
    plt.annotate("a={}".format(a), (1, 1.6))
    plt.annotate("b={}".format(b), (1, 1.3))
plt.show()


tau = Gamma(a=1, b=1)
model = Gaussian(mu=0, tau=tau)
print(model)

model.fit(np.random.normal(scale=1.414, size=100))
print(model)

# student's t-distribution
X = np.random.normal(size=20)
X = np.concatenate([X, np.random.normal(loc=20., size=3)])
plt.hist(X.ravel(), bins=50, normed=1., label="samples")

students_t = StudentsT()
gaussian = Gaussian()

gaussian.fit(X)
students_t.fit(X)

print(gaussian)
print(students_t)

x = np.linspace(-5, 25, 1000)
plt.plot(x, students_t.pdf(x), label="student's t", linewidth=2)
plt.plot(x, gaussian.pdf(x), label="gaussian", linewidth=2)
plt.legend()
plt.show()

# mixture of gaussians

# 既然是高斯混合模型， 那就要多用几次normal，生成正态分布， 再拟合，自然是多个正态分布的混合喽
x1 = np.random.normal(size=(100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size=(100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size=(100, 2))
x3 += np.array([0, 5])
X = np.vstack((x1, x2, x3))

model = MultivariateGaussianMixture(n_components=3)
model.fit(X)
print(model)

x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
probs = model.pdf(X_test)
Probs = probs.reshape(100, 100)
plt.scatter(X[:, 0], X[:, 1])
plt.contour(x_test, y_test, Probs)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
