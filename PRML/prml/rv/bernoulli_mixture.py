import numpy as np
from scipy.misc import logsumexp
from prml.rv.rv import RandomVariable

class BernoulliMixture(RandomVariable):
    """
    p(x|pi,mu)
    = sum_k pi_k mu_k^x (1 - mu_k)^(1 - x)  http://blog.csdn.net/gyarenas/article/details/71125332可以看看公式， 其中π是混合模型各部分系数
    """

    def __init__(self, n_components=3, mu=None, coef=None):
        """
        construct mixture of Bernoulli 构建混合伯努利模型
        Parameters
        ----------
        n_components : int
            number of bernoulli component # 伯努利模型的数量
        mu : (n_components, ndim) np.ndarray
            probability of value 1 for each component # 每一个部分值为1的概率
        coef : (n_components,) np.ndarray
            mixing coefficients 混合系数
        """
        super().__init__()
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.mu = mu
        self.coef = coef

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 2
            assert np.size(mu, 0) == self.n_components
            assert (mu >= 0.).all() and (mu <= 1.).all()
            self.ndim = np.size(mu, 1)
            self.parameter["mu"] = mu
        else:
            assert mu is None
            self.parameter["mu"] = None

    @property
    def coef(self):
        return self.parameter["coef"]

    @coef.setter
    def coef(self, coef):
        if isinstance(coef, np.ndarray):
            assert coef.ndim == 1
            assert np.allclose(coef.sum(), 1)
            self.parameter["coef"] = coef
        else:
            assert coef is None
            self.parameter["coef"] = np.ones(self.n_components) / self.n_components

    def _log_bernoulli(self, X):
        np.clip(self.mu, 1e-10, 1 - 1e-10, out=self.mu) # 把mu限制在0 1 之间 ，但是因为计算机的原因由1e-10代替0
        return (
            X[:, None, :] * np.log(self.mu)
            + (1 - X[:, None, :]) * np.log(1 - self.mu)    # ∑x*log(u) + (1 - x) * log(1-u)
        ).sum(axis=-1)

    def _fit(self, X):
        self.mu = np.random.uniform(0.25, 0.75, size=(self.n_components, np.size(X, 1))) # 用正态分布产生mu值， 
        params = np.hstack((self.mu.ravel(), self.coef.ravel())) # mu, coef就是参数，要把它们串在一起
        while True:
            resp = self._expectation(X)
            self._maximization(X, resp)
            new_params = np.hstack((self.mu.ravel(), self.coef.ravel())) # 用EM算法估计参数 并且采用迭代的方式
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        log_resps = np.log(self.coef) + self._log_bernoulli(X)   # 看看上面的链接，就知道这个公式是怎么回事了 而且也有助于理解HMM那一节的内容
        log_resps -= logsumexp(log_resps, axis=-1)[:, None]
        resps = np.exp(log_resps)
        return resps   # 负对数似然函数  ，属于哪个类的概率

    def _maximization(self, X, resp):
        Nk = np.sum(resp, axis=0)
        self.coef = Nk / len(X)
        self.mu = (X.T @ resp / Nk).T

    def classify(self, X):
        """
        classify input
        max_z p(z|x, theta)
        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input
        Returns
        -------
        output : (sample_size,) ndarray
            corresponding cluster index 相关簇索引
        """
        return np.argmax(self.classify_proba(X), axis=1)  # 输出属于哪个类 概率最大的那个z的后验

    def classfiy_proba(self, X):
        """
        posterior probability of cluster
        p(z|x,theta)
        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input
        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior probability of cluster    # 簇（类别）的后验概率，就是给定输入数据x， 和参数θ， 确定所属类别z
        """
        return self._expectation(X)

# 协方差矩阵的拟合，要借助gram矩阵，因为两者比较相似，可能就差一点小数。
