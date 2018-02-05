# https://max.book118.com/html/2015/0610/18747834.shtm 看完这个链接，全明白了
"""
说到ARD，不得不提RVM， 它是类似于SVM的一种分类方法，但是基于贝叶斯框架，要借助概率的思想思考问题。
RVM有着与 SVM类似的判别公式， 如 样本到分类面的距离f：
f(x, w) = ∑wn * K(x, xn) + w0 = W * Φ
只有参数w， 那么我们就假设w的先验服从高斯分布。借用某种方法可以移除不相关的点。增加稀疏性，只保留少数相关的点
"""
import numpy as np

class RelevanceVectorRegressor(object):
	def __init__(self, kernel, alpha = 1., beta = 1.):
		"""
		construct relevance vector regressor 构建相关向量回归

		Parameters
		----------
		kernel : Kernel 
			kernel function to compute components of feature vectors  计算特征向量的乘积的核函数
		alpha : float
			initial precision of prior weights distribution 先验权重分布的初始精度
		beta : float
			precision of observation 观测精度
		"""
		self.kernel = kernel
		self.alpha = alpha
		self.beta = beta

	def fit(self, X, t, iter_max = 1000):
		"""
		maximize evidence with respect to hyperparameter 最大化与参数相关的证据

		Parameters
		----------
		X : (N, n_features) ndarray
			relevance vector 相关向量
		t : (N, ) ndarray
			corresponding target 相关的对应的目标
		alpha : (N, ) ndarray
			hyperparameter for each weight or training sample 每一个权重或者训练样本的超参数
		cov : (N, N) ndarray
			covariance matrix of weight 权重的协方差矩阵
		mean : (N, ) ndarray
			mean of each weight 每一个权重的均值
		"""
		if X.ndim == 1:
			X = X[:, None]
		assert X.ndim == 2
		assert t.ndim == 1
		N = len(t)
		Phi = self.kernel(X, X)
		self.alpha = np.zeros(N) + self.alpha # 权重的精度，每一个权重都应该有一个，所以维度应该和样本个数一样
		for _ in range(iter_max):
			params = np.hstack([self.alpha, self.beta])
			precision = np.diag(self.alpha) + self.beta * Phi.T @ Phi # 海塞因矩阵13
			covariance = np.linalg.inv(precision) # P22
			mean = self.beta * covariance @ Phi.T @ t
			gamma = 1 - self.alpha * np.diag(covariance) # P22--(18)
			self.alpha = gamma / np.square(mean) # P22--(18)
			np.clip(self.alpha, 0, 1e10, out = self.alpha)
			self.beta = (N - np.sum(gamma)) / np.sum((t - Phi.dot(mean)) ** 2)
			if np.allclose(params, np.hstack([self.alpha, self.beta])):
				break
		mask = self.alpha < 1e9
		self.X = X[mask]
		self.t = t[mask]
		self.alpha = self.alpha[mask]
		Phi = self.kernel(self.X, self.X)
		precision = np.diag(self.alpha) + self.beta * Phi.T @ Phi
		self.covariance = np.linalg.inv(precision)
		self.mean = self.beta * self.covariance @ Phi.T @ self.t

	def predict(self, X, with_error = True):
		"""
		predict output with this model  用模型预测输出

		Parameters
		----------
		X : (sample_size, n_features)
			input
		with_error : bool
			if true, predict with standard deviation of the outputs 用输出的标准偏差预测

		Returns
		-------
		mean : (sample_size, ) ndarray
			mean of predictive distribution 预测分布的均值
		std : (sample_size, ) ndarray
			standard deviation of predictive distribution 预测分布的标准差
		"""
		if X.ndim == 1:
			X = X[:, None]
		assert X.ndim == 2
		phi = self.kernel(X, self.X)
		mean = phi @ self.mean # w * K(X, X)
		if with_error:
			var = 1 / self.beta + np.sum(phi @ self.covariance * phi, axis = 1) # 这个方差是怎么求的呢？
			return mean, np.sqrt(var) # 如果有要求， 就用标准差来预测； 没有要求， 就只用均值
		return mean
