import numpy as np 
from prml.linear.regressor import Regressor 

class RidgeRegressor(Regressor):
	"""
	ridge regression model
	w* = argmin |t - X @ w| + a * |w|_2^2
	"""

	def __init__(self, alpha = 1.):
		self.alpha = alpha

	def _fit(self, X, t):
		eye = np.eye(np.size(X, 1))
		self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t) #残差平方和对β求偏导并置为零，可以得到岭回归的解 β^ridge=(XTX+λI)−1XTy 
	# http://blog.csdn.net/u014664226/article/details/52121865,   岭回归的求解

	def _predict(self, X):
		y = X @ self.w
		return y

# 代码写出来就这么几行，其实需要推导
