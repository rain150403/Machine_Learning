# 老样子，一开始， 先是设置参数， mu, tau, dof, 然后是设置性质维数ndim, 尺寸size, 形状shape
# 拟合的时候，求参数用的是EM算法， 当然，必不可少的pdf()函数。


import numpy as np
from scipy.special import gamma, digamma
from prml.rv.rv import RandomVariable


class StudentsT(RandomVariable):
    """
    Student's t-distribution
    p(x|mu, tau, dof)
    = (1 + tau * (x - mu)^2 / dof)^-(D + dof)/2 / const.
    """

    def __init__(self, mu=None, tau=None, dof=None):
        super().__init__()
        self.mu = mu
        self.tau = tau
        self.dof = dof

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            self.parameter["mu"] = np.array(mu)
        elif isinstance(mu, np.ndarray):
            self.parameter["mu"] = mu
        else:
            assert mu is None
            self.parameter["mu"] = None

    @property
    def tau(self):
        return self.parameter["tau"]

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, (int, float, np.number)):
            tau = np.array(tau)
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
        elif isinstance(tau, np.ndarray):
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
        else:
            assert tau is None
            self.parameter["tau"] = None

    @property
    def dof(self):
        return self.parameter["dof"]

    @dof.setter
    def dof(self, dof):
        if isinstance(dof, (int, float, np.number)):
            self.parameter["dof"] = dof
        else:
            assert dof is None
            self.parameter["dof"] = None

    @property
    def ndim(self):
        if hasattr(self.mu, "ndim"):
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mu, "size"):
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape
        else:
            return None
    """
    首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，
    对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。
    """

    # 这里求参数好像很简单，又好像不简单，好像是用的期望最大化？
    # 根据输入X求参数， 利用期望最大化求参数，如果两者相同， 就好了，如果不同，就选择利用EM算法求得的参数。

    def _fit(self, X, learning_rate=0.01):
        self.mu = np.mean(X, axis=0)
        self.tau = 1 / np.var(X, axis=0)
        self.dof = 1
        params = np.hstack(
            (self.mu.ravel(),
             self.tau.ravel(),
             self.dof)
        )
        while True:
            E_eta, E_lneta = self._expectation(X)
            self._maximization(X, E_eta, E_lneta, learning_rate)
            new_params = np.hstack(
                (self.mu.ravel(),
                 self.tau.ravel(),
                 self.dof)
            )
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        d = X - self.mu
        a = 0.5 * (self.dof + 1)
        b = 0.5 * (self.dof + self.tau * d ** 2)
        E_eta = a / b
        E_lneta = digamma(a) - np.log(b) # 伽玛函数的对数的导数称为Digamma函数
        return E_eta, E_lneta

    # 这应该是EM算法， 公式比较复杂，以后慢慢看
    def _maximization(self, X, E_eta, E_lneta, learning_rate):
        self.mu = np.sum(E_eta * X, axis=0) / np.sum(E_eta, axis=0)
        d = X - self.mu
        self.tau = 1 / np.mean(E_eta * d ** 2, axis=0)
        N = len(X)
        self.dof += learning_rate * 0.5 * (
            N * np.log(0.5 * self.dof) + N
            - N * digamma(0.5 * self.dof)
            + np.sum(E_lneta - E_eta, axis=0)
        )

    # 就是学生t分布的那个公式 ，   55下面那个
    def _pdf(self, X):
        d = X - self.mu
        D_sq = self.tau * d ** 2
        return (
            gamma(0.5 * (self.dof + 1))
            * self.tau ** 0.5
            * (1 + D_sq / self.dof) ** (-0.5 * (1 + self.dof))
            / gamma(self.dof * 0.5)
            / (np.pi * self.dof) ** 0.5
        )
