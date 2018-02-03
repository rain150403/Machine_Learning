# sampling methods
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline 

from prml.rv import Gaussian, Uniform 
from prml.sampling import metropolis, metropolis_hastings, rejection_sampling, sir

np.random.seed(1234)

# rejection sampling
def func(x):
	return np.exp(-x**2) + 3 * np.exp(-(x-3) ** 2)
x = np.linspace(-5, 10, 100)
rv = Gaussian(mu = np.array([2.]), var = np.array([2.]))
plt.plot(x, func(x), label = r"$\tilde{p}(z)$")
plt.plot(x, 15 * rv.pdf(x), label = r"$kq(z)$")
plt.fill_between(x, func(x), 15 * rv.pdf(x), color = "gray")
plt.legend(fontsize = 15)
plt.show()

# 要采样的是func， 一个不了解的分布， 高斯采样容易， 那么就借助高斯分布

samples = rejection_sampling(func, rv, k = 15, n = 100)
plt.plot(x, func(x), label = r"$\tilde{p}(z)$")
plt.hist(samples, normed = True, alpha = 0.2)  # 也就是采样的直方图， 还挺符合func这个分布的
plt.scatter(samples, np.random.normal(scale = .03, size = (100, 1)), s = 5, label = "samples")
plt.legend(fontsize = 15)
plt.show()

####################################################################

# sampling-importance-resampling
samples = sir(func, rv, n = 100)
plt.plot(x, func(x), label = r"$\tilde{p}(z)$")
plt.hist(samples, normed = True, alpha = 0.2)
plt.scatter(samples, np.random.normal(scale = .03, size = (100, 1)), s = 5, label = "samples")
plt.legend(fontsize = 15)
plt.show()

#######################################################################

# markov chain monte carlo
samples = metropolis(func, Gaussian(mu = np.zeros(1), var = np.ones(1)), n = 100, downsample = 10)
plt.plot(x, func(x), label = r"$\tilde{p}(z)$")
plt.hist(samples, normed = True, alpha = 0.2)
plt.scatter(samples, np.random.normal(scale = .03, size = (100, 1)), s = 5, label = "samples")
plt.legend(fontsize = 15)
plt.show()

######################################################################

# the metropolis hastings algorithm
samples = metropolis_hastings(func, Gaussian(mu = np.ones(1), var = np.ones(1)), n = 100, downsample = 10)
plt.plot(x, func(x), label = r"$\tilde{p}(z)$")
plt.hist(samples, normed = True, alpha = 0.2)
plt.scatter(samples, np.random.normal(scale = .03, size = (100, 1)), s = 5, label = "samples")
plt.legend(fontsize = 15)
plt.show()

# 就是用四种方法对同一个func做采样，每一种方法都是借用高斯分布， 采样效果越往后越好
