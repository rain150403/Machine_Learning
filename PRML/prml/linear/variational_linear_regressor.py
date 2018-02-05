# 蝴蝶过山门， 轻舟过重山

# 这里还牵扯一个故事：一个高僧，晚年在一道宏伟的山门上，看到一只弱不禁风的蝴蝶摇摇摆摆就飞过去了。那一刹，他顿悟了人生的轻盈与沉重。我们以为自己爱得死去活来，没法放弃；可是，就一个微小的关节眼，你会突然清醒过来。

# 总有蝴蝶过沧海

#########################################################################

import numpy as np
from prml.linear.regressor import Regressor 

class VariationalLinearRegressor(Regressor):
	"""
	variational bayesian estimation the parameters
	变分贝叶斯估计参数
	p(w, alpha|X, t)
	~ q(w)q(alpha)
	= N(w|w_mean, w_var)Gamma(alpha|a, b)
	就是根据输入数据和标签，估计参数w， alpha，
	这两个参数是独立的，互不相干，
	w服从正态分布，有均值和方差
	alpha服从gamma分布， 有a， b

	Attributes
	----------
	a : float
		a parameter of variational posterior gamma distribution
		变分后验gamma分布的参数A

	b : float
		another parameter of variational posterior gamma distribution
		变分后验gamma分布的另一个参数B

	w_mean : (n_features, ) ndarray
		mean of variational posterior gaussian distribution
		变分后验高斯分布的均值
	w_var : (n_features, n_features) ndarray
		variance of variational posterior gaussian distribution
		变分后验高斯分布的方差

	n_iter : int
		number of iterations performed
		执行迭代的数量
	"""

	def __init__(self, beta = 1., a0 = 1., b0 = 1.):
		"""
		construct variational linear regressor
		构建变分线性回归

		Parameters
		----------
		beta : float
			precision of observation noise 观测噪音的精度
		a0 : float
			a parameter of prior gamma distribution 先验gamma分布的参数a0
			Gamma(alpha|a0, b0)
		b0 : float
			another parameter of prior gamma distribution 先验gamma分布的另一个参数b0
			Gamma(alpha|a0, b0)
		"""
		self.beta = beta
		self.a0 = a0
		self.b0 = b0

	def _fit(self, X, t, iter_max = 100):
		assert X.ndim == 2
		assert t.ndim == 1
		self.a = self.a0 + 0.5 * np.size(X, 1)
		self.b = b0
		I = np.eye(np.size(X, 1))
		for i in range(iter_max):
			param = self.b
			self.w_var = np.linalg.inv(   # https://max.book118.com/html/2015/0610/18747834.shtm  这里的求方差的公式
				self.a * I / self.b
				+ self.beta * X.T @ X)
			self.w_mean = self.beta * self.w_var @ X.T @ t  # 记住吧，求均值都这么求
			self.b = self.b0 + 0.5 * (
				np.sum(self.w_mean**2)
				+ np.trace(self.w_var))
			if np.allclose(self.b, param):
				break
		self.n_iter = i + 1

	def _predict(self, X, return_std = False):
		assert X.ndim == 2
		y = X @ self.w_mean
		if return_std:
			y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis = 1) # 记住吧，求方差都这么求
			y_std = np.sqrt(y_var)
			return y, y_std # 才知道为什么要求这个，为了制订y的界限
		return y

# 拟合就是用各种方法求参数， 预测就是求y
