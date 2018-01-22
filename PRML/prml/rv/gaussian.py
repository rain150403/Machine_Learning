# 关于参数估计的方法，也要输入X服从一定的分布（或者说是参数服从一定的分布），然后借此选择对应的方法

import numpy as np 
from prml.rv.rv import RandomVariable
from prml.rv.gamma import Gamma 

class Gaussian(RandomVariable):
	"""
	the Gaussian distribution

	p(x|mu, var)
	= exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
	"""

	# 既然tau是由var算出来的， 那怎么还需要传入一个参数呢？
	# mu服从 gaussian 分布， tau服从 gamma分布

	def __init__(self, mu = None, var = None, tau = None):
		super().__init__()
		self.mu = mu
		if var is not None:
			self.var = var
		elif tau is not None:
			self.tau = tau
		else:
			self.var = None
			self.tau = None

	@property
	def mu(self):
	    return self.parameter["mu"]

	@mu.setter
	def mu(self, mu):
		if isinstance(mu, (int, float, np.number)):
			self.parameter["mu"] = np.array(mu)
		elif isinstance(mu, np.ndarray):
			self.parameter["mu"] = mu
		elif isinstance(mu, Gaussian):
			self.parameter["mu"] = mu
		else:
			if mu is not None:
				raise TypeError(f"{type(mu)} is not supported for mu")
			self.parameter["mu"] = None

	@property
	def var(self):
	    return self.parameter["var"]


	@var.setter
	def var(self, var):
		if isinstance(var, (int, float, np.number)):
			assert var > 0 
			var = np.array(var)
			assert var.shape == self.shape
			self.parameter["var"] = var
			self.parameter["tau"] = 1/var
		elif isinstance(var, np.ndarray):
			assert (var > 0).all()
			assert var.shape == self.shape
			self.parameter["var"] = var
			self.parameter["tau"] = 1/var
		else:
			assert var is None
			self.parameter["var"] = None
			self.parameter["tau"] = None

	@property
	def tau(self):
		return self.parameter["tau"]

	@tau.setter
    def tau(self, tau):
        if isinstance(tau, (int, float, np.number)):
            assert tau > 0
            tau = np.array(tau)
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = 1 / tau
        elif isinstance(tau, np.ndarray):
            assert (tau > 0).all()
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = 1 / tau
        elif isinstance(tau, Gamma):
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = None
        else:
            assert tau is None
            self.parameter["tau"] = None
            self.parameter["var"] = None

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
如果mu符合Gaussian分布， 并且tau符合gamma分布，那就没法处理了
如果只有一个服从特定的分布， 那么我们就可以按照服从特定分布的那种贝叶斯估计参数了
如果两个都不服从特定分布，那就用极大似然估计。
"""

    def _fit(self, X):
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        if mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError
        elif mu_is_gaussian:
            self._bayes_mu(X)
        elif tau_is_gamma:
            self._bayes_tau(X)
        else:
            self._ml(X)

# 如此看来，极大似然估计就是整体的均值和方差
	def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

# 在mu服从Gaussian分布， var服从ndarray时，采用map， 但是这里只求得了mu
    def _map(self, X):
        assert isinstance(self.mu, Gaussian)
        assert isinstance(self.var, np.ndarray)
        N = len(X)
        mu = np.mean(X, 0)
        self.mu = (
            (self.tau * self.mu.mu + N * self.mu.tau * mu)
            / (N * self.mu.tau + self.tau)
        )

# mu服从Gaussian分布时， 在这里先求Gaussian分布所需的参数， 然后根据高斯分布求得mu
    def _bayes_mu(self, X):
        N = len(X)
        mu = np.mean(X, 0)
        tau = self.mu.tau + N * self.tau
        self.mu = Gaussian(
            mu=(self.mu.mu * self.mu.tau + N * mu * self.tau) / tau,
            tau=tau
        )
# tau服从gamma分布时， 在这里先求gamma分布所需的参数， 然后根据gamma分布求得tau
    def _bayes_tau(self, X):
        N = len(X)
        var = np.var(X, axis=0)
        a = self.tau.a + 0.5 * N
        b = self.tau.b + 0.5 * N * var
        self.tau = Gamma(a, b)

# 既然在这里直接实现了， 那还要上面的两个参数干嘛
    def _bayes(self, X):
        N = len(X)
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        if mu_is_gaussian and not tau_is_gamma:
            mu = np.mean(X, 0)
            tau = self.mu.tau + N * self.tau
            self.mu = Gaussian(
                mu=(self.mu.mu * self.mu.tau + N * mu * self.tau) / tau,
                tau=tau
            )
        elif not mu_is_gaussian and tau_is_gamma:
            var = np.var(X, axis=0)
            a = self.tau.a + 0.5 * N
            b = self.tau.b + 0.5 * N * var
            self.tau = Gamma(a, b)
        elif mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError
        else:
            raise NotImplementedError


    def _pdf(self, X):
        d = X - self.mu
        return (
            np.exp(-0.5 * self.tau * d ** 2) / np.sqrt(2 * np.pi * self.var)
        )

    def _draw(self, sample_size=1):
        return np.random.normal(
            loc=self.mu,
            scale=np.sqrt(self.var),
            size=(sample_size,) + self.shape
        )
