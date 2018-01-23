import random
import numpy as np

def metropolis(func, rv, n, donwsample = 1):
	"""
	Metropolis algorithm

	Parameters
	----------
	func : callable 可调用对象，函数
		(un)normalized distribution to be sampled from 我们将要采样的分布（标准化或者非标准化分布）
	rv : RandomVariable
		proposal distribution which is symmetric at the origin 在原点对称分布的建议分布
	n : int
		numer of samples to draw 要采样的样本数
	donwsample : int
		downsampling factor 下采样因子

		不知道这样设计的好处在哪里，一开始扩展了抽样样本数，但是后来又去掉了，基本没变？？？

	Returns
	-------
	sample : (n, ndim) ndarray
		generated sample 生成样本
	"""
	x = np.zeros((1, rv.ndim))
	sample = []
	for i in range(n * donwsample):
		x_new = x + rv.draw()
		accept_proba = func(x_new) / func(x)
		if random.random() < accept_proba:
			x = x_new
		if i % donwsample == 0:
			sample.append(x[0])
	sample = np.asarray(sample)
	assert sample.shape == (n, rv.ndim), sample.shape
	return sample 



"""
初始化时间t=1
设置u的值，并初始化初始状态θ(t)=u
重复以下的过程： 
	令t=t+1
	从已知分布q(θ∣θ(t−1))中生成一个候选状态θ(∗)
	计算接受的概率：α=min(1,p(θ(∗))p(θ(t−1))q(θ(t−1)∣θ(∗))q(θ(∗)∣θ(t−1)))
	从均匀分布Uniform(0,1)生成一个随机值a
	如果a⩽α，接受新生成的值：θ(t)=θ(∗)；否则：θ(t)=θ(t−1)
"""
