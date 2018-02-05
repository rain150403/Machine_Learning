import numpy as np
from prml.linear.regressor import Regressor 

class EmpiricalBayesRegressor(Regressor):

	def __init__(self, alpha = 1., beta = 1.):
		self.alpha = alpha
		self.beta = beta

	# 经验贝叶斯，参数的先验是根据数据估计出来的， 所以这里并没有给出w_mean, w_precision
	def _fit(self, X, t, max_iter = 100):
		M = X.T @ X
		eigenvalues = np.linalg.eigvalsh(M) # 计算厄米或实对称矩阵的特征值。Compute the eigenvalues of a Hermitian or real symmetric matrix. hermite矩阵
		# n阶复方阵A的对称单元互为共轭，即A的共轭转置矩阵等于它本身，则A是厄米特矩阵(Hermitian Matrix)。
		eye = np.eye(np.size(X, 1))
		N = len(t)
		# 因为要估计参数，所以，这里有迭代操作
		for _ in range(max_iter):
			params = [self.alpha, self.beta]

			# 这里少了判断，这个公式和前一节是一样的
			w_precision = self.alpha * eye + self.beta * X.T @ X
			w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t) # 这里公式不同

			# 这里的参数估计需要再理解理解？？？
			gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
			self.alpha = float(gamma / np.sum(w_mean**2).clip(min = 1e-10))
			self.beta = float(
				(N - gamma) / np.sum(np.square(t - X @ w_mean))
			)
			if np.allclose(params, [self.alpha, self.beta]):
				break

		self.w_mean = w_mean
		self.w_precision = w_precision
		self.w_cov = np.linalg.inv(w_precision)

	def log_evidence(self, X, t):
		"""
		log evidence function 
		
		Parameters
		----------
		X : ndarray (sample_size, n_features)
			input data

		t : ndarray (sample_size, )
			target data

		Returns
		-------
		output : float
			log evidence
		"""
		M = X.T @ X
		return 0.5 * (
			len(M) * np.log(self.alpha)
			+ len(t) * np.log(self, beta)
			- self.beta * np.square(t - X @ self.w_mean).sum()
			- self.alpha * np.sum(self.w_mean**2)
			- np.linalg.slogdet(self.w_precision)[1] # Compute the sign and (natural) logarithm of the determinant of an array.
			- len(t) * np.log(2 * np.pi)   # PRML ---- P167---3.86
		)

	# 这里跟上一个bayesian regressor一样
	def _predict(self, X, return_std = False, sample_size = None):
		if isinstance(sample_size, int):
			w_sample = np.random.multivariate_normal(
				self.w_mean, self.w_cov, size = sample_size
			)
			y = X @ w_sample.T
			return y
		y = X @ self.w_mean
		if return_std:
			y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis = 1)
			y_std = np.sqrt(y_var)
			return y, y_std
		return y


"""
一个完全的贝叶斯分析包括数据分析、概率模型的构造、先验信息和效应函数的假设以及最后的决策（Lindley，2000）

https://baike.baidu.com/item/%E7%BB%8F%E9%AA%8C%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%96%B9%E6%B3%95/1413243?fr=aladdin
当先验分布未知时，利用历史样本来估计先验分布，并确定贝叶斯决策函数，这样的方法称为经验贝叶斯方法。一个决策函数，
它不仅利用当前样本，还利用历史本来确定先验分布，称这样的决策函数为经验贝叶斯决策函数。

经验贝叶斯方法通常可分为参数经验贝叶斯方法和非参数经验贝叶斯方法两种。
设样本的条件密度为  ，参数  的先验分布  未知。记样本的边缘密度为

则  也未知。
若假设  的先验分布属于一己知的参数族，记为  ，则样本的边缘密度可写为  。那么，基于独立同分布  的历史样本  ，
利用经典统计方法可以给出λ 的估计  ，进而得到先验分布的估计  ，并以其贝叶斯解为经验贝叶斯决策函数。这种方法称为参数经验贝叶斯方法。
若决策问题的贝叶斯解可以表示为  ，其中  和  是已知的函数，则可以用服从  的独立样本来估计  ，进而估计贝叶斯解  并作为决策问题的经验贝叶斯决策函数。
这种方法称为非参数经验贝叶斯方法。
经验贝叶斯方法的关键是要有历史样本。[1] 
"""
