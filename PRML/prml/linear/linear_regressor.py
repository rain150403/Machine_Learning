import numpy as np
from prml.linear.regressor import Regressor


class LinearRegressor(Regressor):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def _fit(self, X, t):
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t)) # 就按照上面的两个公式，直接求就行

    def _predict(self, X, return_std=False):
        y = X @ self.w # 按公式来
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
