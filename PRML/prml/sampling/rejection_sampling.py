import random
import numpy as np

def rejection_sampling(func, rv, k, n):
	"""
	perform rejection sampling n times 执行拒绝采样n次

	Parameters
	----------
	func : callable
		(un)normalized distribution to be sampled from 将要采样的分布
	rv : RandomVariable
		distribution to generate sample  用于生成样本的分布
	k : float
		constant to be multiplied with the distribution 高斯分布要乘的系数k
	n : int 
		number of samples to draw 要采多少个样本n

	Returns
	-------
	sample : (n, ndim) ndarray
		generated sample
	"""
	assert hasattr(rv, "draw"), "the distribution has no method to draw random samples"
	sample = []
	while len(sample) < n:
		sample_candidate = rv.draw()  # 只要样本没有达到数量， 就从高斯分布中采集
		accept_proba = func(sample_candidate) / (k * rv.pdf(sample_candidate))
		if random.random() < accept_proba:
			sample.append(sample_candidate[0])
	sample = np.asarray(sample)
	assert sample.shape == (n, rv.ndim), sample.shape
	return sample

# 随机产生一个概率值， 如果小于接受率， 就接受
