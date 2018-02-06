# 协方差矩阵的拟合，要借助gram矩阵，因为两者比较相似，可能就差一点小数。竟被误导了，其实没有什么协方差矩阵的事情，
# 就是取了这么一个名字，知道要求这个矩阵就行了

# http://www.cnblogs.com/little-YTMM/p/5399532.html
import numpy as np 

class GaussianProcessRegressor(object):
	def __init__(self, kernel, beta = 1.):
		"""
		construct gaussian process regressor 构建高斯过程回归

		Parameters
		----------
		kernel
			kernel function
		beta : float
			precision parameter of observation noise 观测噪声的精度参数：beta
		"""
		self.kernel = kernel
		self.beta = beta

	def fit(self, X, t, iter_max = 0, learning_rate = 0.1):
		"""
		maximum likelihood estimation of parameters in kernel function 核函数中的参数的极大似然估计

		Parameter
		---------
		X : ndarray (sample_size, n_features)
			input 
		t : ndarray (sample_size, )
			corresponding target
		iter_max : int
			maximum number of itertools updating hyperparameters
		learning_rate : float
			updation coefficient  更新系数

		Attributes
		----------
		covariance : ndarray (sample_size, sample_size)
			variance covariance matrix of gaussian process 高斯过程的协方差矩阵
		precision : ndarray (sample_size, sample_size)
			precision matrix of gaussian process 高斯过程的精度矩阵

		Returns
		-------
		log_likelihood_list : list 
			list of log likelihood value at each iteration 每次迭代的对数似然值的list
		"""

		if X.ndim == 1:
			X = X[:, None]
		log_likelihood_list = [-np.Inf]
		self.X = X
		self.t = t
		I = np.eye(len(X))
		Gram = self.kernel(X, X)
		self.covariance = Gram + I / self.beta   # PRML P307----6.62
		self.precision = np.linalg.inv(self.covariance) 
		for i in range(iter_max):
			gradients = self.kernel.derivatives(X, X)
			updates = np.array(
				[-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])

			for j in range(iter_max):
				self.kernel.update_parameters(learning_rate * updates)
				Gram = self.kernel(X, X)
				self.covariance = Gram + I / self.beta
				self.precision = np.linalg.inv(self.covariance)
				log_like = self.log_likelihood()
				if log_like > log_likelihood_list[-1]:
					log_likelihood_list.append(log_like)
					break
				else:
					self.kernel.update_parameters(-learning_rate * updates)
					learning_rate *= 0.9

		log_likelihood_list.pop(0)
		return log_likelihood_list

	def log_likelihood(self):
		return -0.5 * (
			np.linalg.slogdet(self.covariance)[1] 
			+ self.t @ self.precision @ self.t
			+ len(self.t) * np.log( 2 * np.pi))  # PRML---P311--6.69

	def predict(self, X, with_error = False):
		"""
		mean of the gaussian process 高斯过程的均值

		Parameters
		----------
		X : ndarray (sample_size, n_features)
			input

		Returns
		-------
		mean : ndarray (sample_size, )
			predictions of corresponding inputs 相关输入的预测---均值
		"""
		if X.ndim == 1:
			X = X[:, None]
		K = self.kernel(X, self.X)
		mean = K @ self.precision @ self.t	# PRML --- P203 --6.66
		if with_error:
			var = (self.kernel(X, X, False) + 1 / self.beta - np.sum(K @ self.precision * K, axis = 1))	# PRML --- P203 --6.67
			return mean.raval(), np.sqrt(var.ravel())
		return mean
