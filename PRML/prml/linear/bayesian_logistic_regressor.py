# PRML --- P217
class BayesianLogisticRegressor(LogisticRegressor):
	"""
	Logistic regression model
	w ~ Gaussian(0, a^(-1)I)
	y = sigmoid(X @ w)
	t ~ Bernoulli(t|y)
	"""
	def __init__(self, alpha = 1.):
		self.alpha = alpha

	def _fit(self, X, t, max_iter = 100):
		self._check_binary(t)
		w = np.zeros(np.size(X, 1))
		eye = np.eye(np.size(X, 1))
		self.w_mean = np.copy(w)
		self.w_precision = self.alpha * eye
		for _ in range(max_iter):
			w_prev = np.copy(w)
			y = self._sigmoid(X @ w)
			grad = X.T @ (y - t) + self.w_precision @ (w - self.w_mean)
			hessian = (X.T * y * (1 - y)) @ X + self.w_precision # PRML ---- P218 --- 4.143 # 是一个多元函数的二阶偏导数构成的方阵, 前面是海森矩阵的正常公式，后面怎么又加了一个w_precision就是个问题，近似？
			try:
				w -= np.linalg.solve(hessian, grad)
			except np.linalg.LinAlgError:
				break
			if np.allclose(w, w_prev):
				break
		self.w_mean = w
		self.w_precision = hessian

	def _proba(self, X):
		mu_a = X @ self.w_mean
		var_a = np.sum(np.linalg.solve(self.w_precision, X.T).T * X, axis = 1)
		y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))  # PRML --- P220---- 4.153
		return y
