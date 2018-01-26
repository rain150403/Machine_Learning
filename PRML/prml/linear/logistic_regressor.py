import numpy as np
from prml.linear.classifier import Classifier


class LogisticRegressor(Classifier):
    """
    Logistic regression model
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def _fit(self, X, t, max_iter=100):
        self._check_binary(t)
        w = np.zeros(np.size(X, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t)
            hessian = (X.T * y * (1 - y)) @ X
            try:
                w -= np.linalg.solve(hessian, grad) # 这里就是用海森矩阵解决logistic回归问题 http://blog.csdn.net/u012526120/article/details/48897135
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _proba(self, X):
        y = self._sigmoid(X @ self.w)
        return y

    def _classify(self, X, threshold=0.5):
        proba = self._proba(X)
        label = (proba > threshold).astype(np.int)
        return label

"""
模型：
h(x) = ∑wi*xi = W*X
损失函数：
J(w) = 1/2∑(h(xj) - yj) ^ 2


解析解：
w = (X.T * X)^-1 * X.T * Y
梯度下降解
gradient = X.T ` (Y - h(X)) 对损失函数求导即可
w_i+1 = w_i + gradient * alpha

http://www.cnblogs.com/Finley/p/5325417.html
"""
