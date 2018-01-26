import numpy as np
from prml.linear.regressor import Regressor 

class BayesianRegressor(Regressor):
	"""
	Bayesian regression model
	w ~ N(w|0, alpha^(-1)I) 满足高斯分布的参数w的先验分布
	y = X @ w
	t ~ N(t|X @ w, beta^(-1)) 线性模型的概率表示
	"""

	def __init__(self, alpha = 1., beta = 1.):
		self.alpha = alpha
		self.beta = beta
		self.w_mean = None
		self.w_precision = None

	def _fit(self, X, t):
		if self.w_mean is not None:
			mean_prev = self.w_mean
		else:
			mean_prev = np.zeros(np.size(X, 1))
		if self.w_precision is not None:
			precision_prev = self.w_precision
		else:
			precision_prev = self.alpha * np.eye(np.size(X, 1))
		w_precision = precision_prev + self.beta * X.T @ X # SN^-1
		w_mean = np.linalg.solve(
			w_precision, 
			precision_prev @ mean_prev + self.beta * X.T @ t # MN
		) # 在这个网页的postier图片里， http://www.52nlp.cn/prml%E8%AF%BB%E4%B9%A6%E4%BC%9A%E7%AC%AC%E4%B8%89%E7%AB%A0-linear-models-for-regression
		self.w_mean = w_mean
		self.w_precision = w_precision
		self.w_cov = np.linalg.inv(self.w_precision)

	def _predict(self, X, return_std = False, sample_size = None):
		if isinstance(sample_size, int):
			w_sample = np.random.mulvariate_normal(
				self.w_mean, self.w_cov, size = sample_size
			) # 多元正态分布 Draw random samples from a multivariate normal distribution.
			y = X @ w_sample.T
			return y
		y = X @ self.w_mean
		if return_std:
			y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis = 1)
			y_std = np.sqrt(y_var)
			return y, y_std
		return y

# 代码实现就是直奔主题，根据给定的值，列出你的参数计算公式，就是拟合fit
# 预测就是利用求得的值，去计算y = x * w
