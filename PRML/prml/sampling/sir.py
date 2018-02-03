import numpy as np

def sir(func, rv, n):
	"""
	sampling-importance-resampling

	Parameters
	----------
	func : callable
		(un)normalized distribution to be sampled from
	rv : RandomVariable
		distribution to generate sample
	n : int
		number of samples to draw

	Returns
	-------
	sample : (n, ndim) ndarray
		generated sample
	"""
	assert hasattr(rv, "draw"), "the distribution has no method to draw random samples"
	sample_candidate = rv.draw(n * 10)
	weight = np.squeeze(func(sample_candidate) / rv.pdf(sample_candidate)) # Remove single-dimensional entries from the shape of an array. np.squeeze: 返回一个把长度为 1 的维度去掉的 ndarray
	assert weight.shape == (n * 10, ), weight.shape
	weight /= np.sum(weight)
	index = np.random.choice(n * 10, n, p = weight)
	sample = sample_candidate[index]
	return sample
# 按权重选择样本， 而权重有自己的计算方式
