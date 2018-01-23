import numpy as np 
from prml.clusterings import KMeans
from prml.rv.rv import RandomVariable

class MultivariateGaussianMixture(RandomVariable):
	"""
	p(x|mu, L, pi(coef))
	= sum_k pi_k N(x|mu_k, L_k^-1)
	"""

	def __init__(self, n_components, mu = None, cov = None, tau = None, coef = None):
		"""
		construct mixture of Gaussians 

		Parameters
		----------
		n_components : int
            number of gaussian component
        mu : (n_components, ndim) np.ndarray
            mean parameter of each gaussian component
        cov : (n_components, ndim, ndim) np.ndarray
            variance parameter of each gaussian component
        tau : (n_components, ndim, ndim) np.ndarray
            precision parameter of each gaussian component
        coef : (n_components,) np.ndarray
            mixing coefficients 混合系数
        """
        super().__init__()
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.mu = mu
        if cov is not None and tau is not None:
            raise ValueError("Cannot assign both cov and tau at a time")
        elif cov is not None:
            self.cov = cov
        elif tau is not None:
            self.tau = tau
        else:
            self.cov = None
            self.tau = None
        self.coef = coef

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 2
            assert np.size(mu, 0) == self.n_components
            self.ndim = np.size(mu, 1)
            self.parameter["mu"] = mu
        elif mu is None:
            self.parameter["mu"] = None
        else:
            raise TypeError("mu must be either np.ndarray or None")

    @property
    def cov(self):
        return self.parameter["cov"]

    @cov.setter
    def cov(self, cov):
        if isinstance(cov, np.ndarray):
            assert cov.shape == (self.n_components, self.ndim, self.ndim)
            self._tau = np.linalg.inv(cov)
            self.parameter["cov"] = cov
        elif cov is None:
            self.parameter["cov"] = None
            self._tau = None
        else:
            raise TypeError("cov must be either np.ndarray or None")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, np.ndarray):
            assert tau.shape == (self.n_components, self.ndim, self.ndim)
            self.parameter["cov"] = np.linalg.inv(tau)
            self._tau = tau
        elif tau is None:
            self.parameter["cov"] = None
            self._tau = None
        else:
            raise TypeError("tau must be either np.ndarray or None")

    @property
    def coef(self):
        return self.parameter["coef"]
    
    # 就是每个component所占的比重
    @coef.setter
    def coef(self, coef):
        if isinstance(coef, np.ndarray):
            assert coef.ndim == 1
            if np.isnan(coef).any():
                self.parameter["coef"] = np.ones(self.n_components) / self.n_components
            elif not np.allclose(coef.sum(), 1):
                raise ValueError(f"sum of coef must be equal to 1 {coef}")
            self.parameter["coef"] = coef
        elif coef is None:
            self.parameter["coef"] = None
        else:
            raise TypeError("coef must be either np.ndarray or None")

    # 这里只设置了shape形状，其它两个都没设定
    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape[1:]
        else:
            return None

    # 就是Gaussian公式，只是求和麻烦一些
    def _gauss(self, X):
    	d = X[:, None, :] - self.mu
    	D_sq = np.sum(np.einsum('nki, kij -> nkj', d, self.cov) * d, -1) # 在操作数上计算爱因斯坦求和约定 ，所谓Einstein约定求和就是略去求和式中的求和号。在此规则中两个相同指标就表示求和，而不管指标是什么字母，有时亦称求和的指标为“哑指标”
    	return (np.exp(-0.5 * D_sq) / np.sqrt(np.linalg.det(self.cov) * (2*np.pi)**self.ndim))


    # 1）聚类的中心就是mu，
    # 2）先求出输入X的cov协方差矩阵， 再按照component选择cov
    # 3）coef 混合系数就是均分， 1/components
    # 依然是先按照上述方法求出来一个params， 在利用期望最大化算法EM求出一个new_params， 两者一样，就好了，不一样，就选择new_params
    def _fit(self, X):
    	cov = np.cov(X.T)
    	kmeans = KMeans(self.n_components)
    	kmeans.fit(X)
    	self.mu = kmeans.centers
    	self.cov = np.array([cov for _ in range(self.n_components)])
    	self.coef = np.ones(self.n_components) / self.n_components
    	params = np.hstack((self.mu.ravel(), self.cov.ravel(), self.coef.ravel()))
    	while True:
    		stats = self._expectation(X)
    		self._maximization(X, stats)
    		new_params = np.hstack((self.mu.ravel(), self.cov.ravel(), self.coef.ravel()))
    		if np.allclose(params, new_params):
    			break
    		else:
    			params = new_params

    def _expectation(self, X):
    	resps = self.coef * self._gauss(X)
    	resps /= resps.sum(axis = -1, keepdims = True)
    	return resps

    def _maximization(self, X, resps):
    	NK = np.sum(resps, axis = 0)
    	self.coef = NK / len(X)
    	self.mu = (X.T @ resps / NK).T
    	d = X[:, None, :] - self.mu
    	self.cov = np.einsum('nki, nkj -> kij', d, d * resps[:, :, None]) / NK[:, None , None]

    def joint_proba(self, X):
    	"""
    	calculate joint probability p(X, Z) 计算联合概率

    	Parameters
    	----------
    	X : (sample_size, n_features) ndarray
    		input data

    	Returns
    	-------
    	joint_proba : (sample_size, n_components) ndarray
    		joint probability of input and component 输入和部分的联合概率，Z是指哪个component
    	"""
    	return self.coef * self._gauss(X) # 系数乘以这个component的Gaussian值就好了

   	# 就是混合模型的公式，联合概率求和
    def _pdf(self, X):
    	joint_proba = self.coef * self._gauss(X) # 如此一来，上面的函数可以省去
    	return np.sum(joint_proba, axis = -1)

    def classify(self, X):
    	"""
    	classify input 分类输入
    	max_z p(z|x, theta)
    	使得概率取最大值的z值

    	Parameters
    	----------
    	X : (sample_size, ndim) ndarray
    		input

    	Returns
    	-------
    	output : (sample_size, ) ndarray
    		corresponding cluster index 相关聚类索引，也就是到底是属于哪个聚类的
    	"""
    	return np.argmax(self.classify_proba(X), axis = 1)

    def classify_proba(self, X):
    	"""
    	posterior probability of cluster
    	聚类的后验概率
    	p(z|x, theta)

    	Parameters
    	----------
    	X : (sample_size, ndim) ndarray
    		input

    	Returns
    	-------
    	output : (sample_size, n_components) ndarray
    		posterior probability of cluster
    	"""
    	return self._expectation(X) # 期望就是最大后验概率？ EM算法的第一步
