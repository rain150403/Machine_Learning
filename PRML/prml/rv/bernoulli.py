import numpy as np 
from prml.rv.rv import RandomVariable
from prml.rv.beta import Beta 


"""
Bernoulli 分布最主要的参数是mu：（每个元素为1的概率），1）要对其进行概率性质的判断，2）如果mu有维数ndim， 尺寸size， 形状shape等性质，那就按照给定的设置。


几个功能函数：
_fit() # 所谓拟合fit就是要求出mu参数， 在这里也就是统计0和1的个数。
_ml() # 一般情况下，mu就是整体的均值， 为此我们要统计0， 1的个数。
_map() # 说到底还是统计0， 1的个数， 在这里利用0， 1的个数求了一个概率值
_bayes() # 要是mu服从beta分布，那就用贝叶斯方法：按每一轴计算0和1的个数，再累加分别得到mu.n_zeros, mu.n_ones
_pdf() # 这里真正的求得了Bernoulli概率
_draw() # 不知道这个draw是什么意思？？？

"""

class Bernoulli(RandomVariable):
	"""
	Bernoulli distribution
	p(x|mu) = mu^x(1-mu)^(1-x)
	"""
	def __init__(self, mu = None):
		"""
		construct Bernoulli distribution

		Parameters
		----------
		mu : np.ndarray or Beta
			probability of value 1 for each element 
		"""
		super().__init__()
		self.mu = mu

	@property
	def mu(self):
	    return self.parameter["mu"]

	@mu.setter
	def mu(self, mu):
		if isinstance(mu, (int, float, np.number)):
			if mu > 1 or mu < 0:
				raise ValueError(f"mu must be in [0, 1], not {mu}")
			self.parameter["mu"] = np.asarray(mu)
		elif isinstance(mu, np.ndarray):
			if(mu > 1).any() or (mu < 0).any():
				raise ValueError("mu must be in [0, 1]")
			self.parameter["mu"] = mu
		elif isinstance(mu, Beta):
			self.parameter["mu"] = mu
		else:
			if mu is not None:
				raise TypeError(f"{type(mu)} is not supported for mu")
			self.parameter["mu"] = None


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

	# 所谓拟合fit就是要求出mu参数， 在这里也就是统计0和1的个数。

	def _fit(self, X):
		if isinstance(self.mu, Beta):
			self._bayes(X)
		elif isinstance(self.mu, RandomVariable):
			raise NotImplementedError # 抛出异常
		else:
			self._ml(X)

	def _ml(self, X):
		n_zeros = np.count_nonzero((X == 0).astype(np.int)) # 数组中非零值的个数，至于为什么要有X == 0？
		n_ones = np.count_nonzero((X == 1).astype(np.int)) # 这两句应该是说数组中1的个数和0的个数吧
		assert X.size == n_zeros + n_ones, ("{X.size} is not equal to {n_zeros} plus {n_ones}") # 要是0的个数加上1的个数就是X的size， 那么就符合我们的要求了
		self.mu = np.mean(X, axis = 0) # 一般情况下，mu就是整体的均值， 为此我们要统计0， 1的个数。

	# 说到底还是统计0， 1的个数，只是不明白为什么要将X.n_ones + mu.n_ones? 本来mu.n_ones不就是根据X.n_ones求得的吗？
	# 在这里利用0， 1的个数求了一个概率值
	def _map(self, X):
		assert isinstance(self.mu, Beta)
		assert X.shape[1:] == self.mu.shape
		n_ones = (X == 1).sum(axis = 0)
		n_zeros = (X == 0).sum(axis = 0)
		assert X.size == n_zeros.sum() + n_ones.sum(), (f"{X.size} is not equal to {n_zeros} plus {n_ones}")
		n_ones = n_ones + self.mu.n_ones
		n_zeros = n_zeros + self.mu.n_zeros
		self.prob = (n_ones - 1) / (n_ones + n_zeros - 2)

	def _bayes(self, X):
		assert isinstance(self.mu, Beta)
		assert X.shape[1:] == self.mu.shape
		n_ones = (X == 1).sum(axis = 0)
		n_zeros = (X == 0).sum(axis = 0)
		assert X.size == n_zeros.sum() + n_ones.sum(), (" input X must only has 0 or 1")
		self.mu.n_zeros += n_zeros
		self.mu.n_ones += n_ones

	# 要是mu服从beta分布，那就用贝叶斯方法：按每一轴计算0和1的个数，再累加分别得到mu.n_zeros, mu.n_ones

	# 这里真正的求得了Bernoulli概率
	def _pdf(self, X):
		assert isinstance(mu, np.ndarray)
		return np.prod(self.mu**X * (1 - self.mu) ** (1 - X))


	# 不知道这个draw是什么意思？？？
	def _draw(self, sample_size = 1):
		if isinstance(self.mu, np.ndarray):
			return (self.mu > np.random.uniform(size = (sample_size, ) + self.shape)).astype(np.int)
		elif isinstance(self.mu, Beta):
			return (self.mu.n_ones / (self.mu.n_ones + self.mu.n_zeros) > np.random.uniform(size = (sample_size, ) + self.shape)).astype(np.int)
		elif isinstance(self.mu, RandomVariable):
			return (self.mu.draw(sample_size) > np.random.uniform(size = (sample_size, ) + self.shape))
	

"""
语法：isinstance（object，type）

作用：来判断一个对象是否是一个已知的类型。 

其第一个参数（object）为对象，第二个参数（type）为类型名(int...)或类型名的一个列表((int,list,float)是一个列表)。其返回值为布尔型（True or flase）。

若对象的类型与参数二的类型相同则返回True。若参数二为一个元组，则若对象类型与元组中类型名之一相同即返回True。
"""	
	
	
