"""
拟合fit：
利用核函数的结果矩阵做GRAM矩阵， gram的每一个元素做微小改动就可以得到协方差矩阵covariance，
协方差矩阵的逆就是precision（拟合的结果）

预测predict：
在做预测的时候，把precision作用于核函数的结果矩阵的每一个元素，t也作用于每一个元素（不知道t是做什么的？）
就得到我们的结果了，再对结果做一个非线性变换（通过激活函数），就是我们最终要的结果。
"""

import numpy as np

class GaussianProcessClassifier(object):

    def __init__(self, kernel, noise_level=1e-4):
        """
        construct gaussian process classifier
        Parameters
        ----------
        kernel
            kernel function to be used to compute Gram matrix
        noise_level : float
            parameter to ensure the matrix to be positive
        """
        self.kernel = kernel
        self.noise_level = noise_level

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t):
        if X.ndim == 1:
            X = X[:, None]
        self.X = X
        self.t = t
        Gram = self.kernel(X, X)
        self.covariance = Gram + np.eye(len(Gram)) * self.noise_level
        self.precision = np.linalg.inv(self.covariance)

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(X, self.X)
        a_mean = K @ self.precision @ self.t
        return self._sigmoid(a_mean)
