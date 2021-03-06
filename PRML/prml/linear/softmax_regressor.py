import numpy as np
from prml.linear.classifier import Classifier


class SoftmaxRegressor(Classifier):
    """
    Softmax regression model
    aka multinomial logistic regression,多项逻辑回归
    multiclass logistic regression, or maximum entropy classifier.最大熵分类
    y = softmax(X @ W)
    t ~ Categorical(t|y)
    """

    def _fit(self, X, t, max_iter=100, learning_rate=0.1):
        self.n_classes = np.max(t) + 1
        T = np.eye(self.n_classes)[t]
        W = np.zeros((np.size(X, 1), self.n_classes))
        for _ in range(max_iter):
            W_prev = np.copy(W)
            y = self._softmax(X @ W)
            grad = X.T @ (y - T) 
            W -= learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.W = W
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
    def _softmax(self, a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def _proba(self, X):
        y = self._softmax(X @ self.W)
        return y

    def _classify(self, X):
        proba = self._proba(X)
        label = np.argmax(proba, axis=-1)
        return label

# 这个代码看来还是很简单的，只是在求导时还有一些小问题
