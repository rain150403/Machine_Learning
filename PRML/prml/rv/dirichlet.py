"""
最主要的是弄明白参数的含义
其次是三种参数估计方法
最次记住公式，表达式

这里的拟合就是借助参数估计方法求出参数

而draw就是根据求得的参数，利用numpy现有的函数，生成该分布，或者说是产生服从该分布的数， 也可以理解成采样，从该分布中采样

并注意区分：二元变量， 多元变量， 高斯分布，  都使用什么分布

最最最重要的是，知道每一种分布是什么样子的，什么情况下会产生这样的分布，它们有什么用
"""

import numpy as np
from scipy.special import gamma
from prml.rv.rv import RandomVariable


class Dirichlet(RandomVariable):
    """
    Dirichlet distribution
    p(mu|alpha)
    = gamma(sum(alpha))
      * prod_k mu_k ^ (alpha_k - 1)
      / gamma(alpha_1) / ... / gamma(alpha_K)
    """

    def __init__(self, alpha):
        """
        construct dirichlet distribution
        Parameters
        ----------
        alpha : (size,) np.ndarray
            pseudo count of each outcome, aka concentration parameter 又叫做集中参数
        http://blog.csdn.net/macanv/article/details/53036095这里有对狄利特雷分布的翻译
        """
        super().__init__()
        self.alpha = alpha

    @property
    def alpha(self):
        return self.parameter["alpha"]

    @alpha.setter
    def alpha(self, alpha):
        assert isinstance(alpha, np.ndarray)
        assert alpha.ndim == 1
        assert (alpha >= 0).all()
        self.parameter["alpha"] = alpha

    @property
    def ndim(self):
        return self.alpha.ndim

    @property
    def size(self):
        return self.alpha.size

    @property
    def shape(self):
        return self.alpha.shape

    def _pdf(self, mu):
        return (
            gamma(self.alpha.sum())
            * np.prod(mu ** (self.alpha - 1), axis=-1)
            / np.prod(gamma(self.alpha), axis=-1)
        )

    def _draw(self, sample_size=1):
        return np.random.dirichlet(self.alpha, sample_size)
