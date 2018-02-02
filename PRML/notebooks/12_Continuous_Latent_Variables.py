# continuous latent variables
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets
# %matplotlib inline

from prml.feature_extractions import Autoencoder, BayesianPCA, PCA 

np.random.seed(1234)

iris = datasets.load_iris()

# principal component analysis
pca = PCA(n_components = 2) # 普通数据，就只有2个components
Z = pca.fit_transform(iris.data)
plt.scatter(Z[:, 0], Z[:, 1], c = iris.target)
plt.gca().set_aspect('equal', adjustable = 'box') # 获取子图
plt.show()

#################### PCA.PY ################
import numpy as np

class PCA(object):
	def __init__(self, n_components):
		"""
		construct principal component analysis

		Parameters
		----------
		n_components : int
			number of components
		"""
		assert isinstance(n_components, int)
		self.n_components = n_components

	def fit(self, X, method = "eigen", iter_max = 100):
		"""
		maximum likelihood estimate of pca parameters
		x ~ \int_z N(x|Wz + mu, sigma^2)N(z|0, I)dz
		PCA参数的极大似然估计

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data
		method : str 
			method to estimate the parameters
			["eigen", "em"]
		iter_max : int
			maximum number of iterations for em algorithm

		Attributes
		----------
		mean : (n_features, ) ndarray
			sample mean of the data 输入数据的样本均值
		W : (n_features, n_components) ndarray
			projection matrix 映射矩阵
		var : float
			variance of observation noise 观测噪音的方差
		C : (n_features, n_features) ndarray
			variance of the marginal dist N(x|mean, C) 边缘分布的方差
		Cinv : (n_features, n_features) ndarray
			precision of the marginal dist N(x|mean, C) 边缘分布的精度
		"""
		method_list = ["eigen", "em"]
		if method not in method_list:
			print("available methods are {}".format(method_list))
		self.mean = np.mean(X, axis = 0)
		getattr(self, method)(X - self.mean, iter_max)

	def eigen(self, X, *arg):
		sample_size, n_features = X.shape
		if sample_size >= n_features:
			cov = np.cov(X, rowvar = False)
			values, vectors = np.linalg.eigh(cov) # Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
			index = n_features - self.n_components
		else:
			cov = np.cov(X)
			values, vectors = np.linalg.eigh(cov)
			vectors = (X.T @ vectors) / np.sqrt(sample_size * values)
			index = sample_size - self.n_components
		self.I = np.eye(self.n_components)
		if index == 0:
			self.var = 0
		else:
			self.var = np.mean(values[:index])

		self.W = vectors[:, index:].dot(np.sqrt(np.diag(values[index:]) - self.var * self.I))
		self.__M = self.W.T @ self.W + self.var * self.I
		self.C =self.W @ self.W.T + self.var * np.eye(n_features)
		if index == 0:
			self.Cinv = np.linalg.inv(self.C)
		else:
			self.Cinv = np.eye(n_features) / np.sqrt(self.var) - self.W @ np.linalg.inv(self.__M) @ self.W.T / self.var

	def em(self, X, iter_max):
		self.I = np.eye(self.n_components)
		self.W = np.eye(np.size(X, 1), self.n_components)
		self.var = 1.
		for i in range(iter_max):
			W = np.copy(self.W)
			stats = self._expectation(X)
			self._maximization(X, *stats)
			if np.allclose(W, self.W):
				break
		self.C = self.W @ self.W.T + self.var * np.eye(np.size(X, 1))
		self.Cinv = np.linalg.inv(self.C)

	def _expectation(self, X):
		self.__M = self.W.T @ self.W + self.var * self.I
		Minv = np.linalg.inv(self.__M)
		Ez = X @ self.W @ Minv
		Ezz = self.var * Minv + Ez[:, :, None] * Ez[:, None, :]
		return Ez, Ezz

	def _maximization(self, X, Ez, Ezz):
		self.W = X.T @ Ez @ np.linalg.inv(np.sum(Ezz, axis = 0))
		self.var = np.mean(
			np.mean(X ** 2, axis = 1)
			- 2 * np.mean(Ez @ self.W.T * X, axis = 1)
			+ np.trace((Ezz @ self.W.T @ self.W).T) / np.size(X, 1))

	def transform(self, X):
		"""
		project input data into latent space
		p(Z|X) = N(Z|(X-mu)WMinv, sigma^-2M)
		把输入数据映射到一个新的空间（隐空间，潜在空间）

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data

		Returns
		-------
		Z : (sample_size, n_components) ndarray
			projected input data
		"""
		return np.linalg.solve(self.__M, ((X - self.mean) @ self.W).T).T

	def fit_transform(self, X, method = "eigen"):
		"""
		perform pca and whiten the input data
		执行PCA操作并白化输入数据

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data

		Returns
		-------
		Z : (sample_size, n_components) ndarray
			projected input data
			输入数据的映射
		"""
		self.fit(X, method)
		return self.transform(X)

	def proba(self, X):
		"""
		the marginal distribution of the observed variable 
		观测变量的边缘分布

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data

		Returns
		-------
		p : (sample_size, ) ndarray
			value of the marginal distribution
		"""
		d = X - self.mean
		return (
			np.exp(-0.5 * np.sum(d @ self.Cinv * d, axis = -1))
			/ np.sqrt(np.linalg.det(self.C))
			/ np.power(2 * np.pi, 0.5 * np.size(X, 1)))

"""
所谓拟合就是估计参数，这里使用了两种方法：极大似然估计，和EM期望最大化算法，然后就是坐标数据映射
"""
##################################################

# PCA for high-dimensional data 所谓的高维数据就是图片，这里components有4个，而且没有transform那一个步骤，这里会涉及映射矩阵
mnist = datasets.fetch_mldata("MNIST original")
mnist3 = mnist.data[np.random.choice(np.where(mnist.target == 3)[0], 200)] # 标签为3的图片随便选200张
pca = PCA(n_components = 4)
pca.fit(mnist3)
plt.subplot(1, 5, 1)
plt.imshow(pca.mean.reshape(28, 28))
plt.axis('off')
for i, w in enumerate(pca.W.T[::-1]): # W 映射矩阵
	plt.subplot(1, 5, i + 2)
	plt.imshow(w.reshape(28, 28))
	plt.axis('off')
plt.show()

# PCA就是先拟合，求出参数，再做坐标变换
# EM algorithm for PCA
pca = PCA(n_components = 2)
Z = pca.fit_transform(iris.data, method = "em")
plt.scatter(Z[:, 0], Z[:, 1], c = iris.target)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()

# bayesian PCA
def create_toy_data(sample_size = 100, ndim_hidden = 1, ndim_observe = 2, std = 1.):
	Z = np.random.normal(size = (sample_size, ndim_hidden))
	mu = np.random.uniform(-5, 5, size = (ndim_observe))
	W = np.random.uniform(-5, 5, (ndim_hidden, ndim_observe))

	X = Z.dot(W) + mu + np.random.normal(scale = std, size = (sample_size, ndim_observe)) # dot 矩阵乘法
	return X

def hinton(matrix, max_weight = None, ax = None):
	"""
	draw hinton diagram for visualizing a weight matrix.
	"""
	ax = ax if ax is not None else plt.gca()

	if not max_weight:
		max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

	ax.patch.set_facecolor('gray')
	ax.set_aspect('equal', 'box')
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())

	for (x, y), w in np.ndenumerate(matrix):
		color = 'white' if w > 0 else 'black'
		size = np.sqrt(np.abs(w) / max_weight)
		rect = plt.Rectangle([y - size / 2, x - size / 2], size, size, facecolor = color, edgecolor = color)
		ax.add_patch(rect)

	ax.autoscale_view()
	ax.invert_yaxis()
	plt.xlim(-0.5, np.size(matrix, 1) - 0.5)
	plt.ylim(-0.5, len(matrix) - 0.5)

X = create_toy_data(sample_size = 100, ndim_hidden = 3, ndim_observe = 10, std = 1.)
pca = PCA(n_components = 9)
pca.fit(X)
bpca = BayesianPCA(n_components = 9)
bpca.fit(X, initial = "eigen")
plt.subplot(1, 2, 1)
plt.title("PCA")
hinton(pca.W)
plt.subplot(1, 2, 2)
plt.title("Bayesian PCA")
hinton(bpca.W)


# autoassociative neural networks
autoencoder = Autoencoder(4, 3, 2)
autoencoder.fit(iris.data, 10000, 1e-3)

Z = autoencoder.transform(iris.data)
plt.scatter(Z[:, 0], Z[:, 1], c = iris.target)
plt.show()

########################## autoencoder.py #################
import numpy as np
from prml import nn 

class Autoencoder(nn.Network):
	def __init__(self, *args):
		self.n_unit = len(args)
		super().__init__()
		for i in range(self.n_unit - 1):
			self.parameter[f"w_encode{i}"] = nn.Parameter(np.random.randn(args[i], args[i+1]))
			self.parameter[f"b_encode{i}"] = nn.Parameter(np.zeros(args[i+1]))
			self.parameter[f"w_decode{i}"] = nn.Parameter(np.random.randn(args[i+1], args[i]))
			self.parameter[f"b_decode{i}"] = nn.Parameter(np.zeros(args[i]))
			
	def transform(self, x):
		h = x 
		for i in range(self.n_unit - 1):
			h = nn.tanh(h @ self.parameter[f"w_encode{i}"] + self.parameter[f"b_encode{i}"])
		return h.value

	# 写他人者亦如他人,我见青山多妩媚，料青山见我应如是
	def forward(self, x):
		h = x
		for i in range(self.n_unit - 1):
			h = nn.tanh(h @ self.parameter[f"w_encode{i}"] + self.parameter[f"b_encode{i}"])
		for i in range(self.n_unit - 2, 0, -1):
			h = nn.tanh(h @ self.parameter[f"w_decode{i}"] + self.parameter[f"b_decode{i}"])
		x_ = h @ self.parameter["w_decode0"] + self.parameter["b_decode0"]
		self.px = nn.random.Gaussian(x_, 1., data = x) # x_是均值， std = 1.

	def fit(self, x, n_iter = 100, learning_rate = 1e-3):
		optimizer = nn.optimizer.Adam(self.parameter, learning_rate)
		for _ in range(n_iter):
			self.clear()
			self.forward(x)
			log_likelihood = self.log_pdf()
			log_likelihood.backward()
			optimizer.update()


########################  bayes pca ######################
import numpy as np
from prml.feature_extractions.pca import PCA

class BayesianPCA(PCA):
	def fit(self, X, iter_max = 100, initial = "random"):
		"""
		empirical bayes estimation of pca parameters
		用经验贝叶斯估计pca的参数

		到底哪里体现贝叶斯了？我倒是看到先是用random或者eigen方法估计参数初始值， 然后再用EM期望最大化算法估计参数的， 而且E步没有变，M步变了
		我想就是先估计参数， 后来又经过迭代方法求得了参数的真正估计值这样吧？就是所谓的先验后验

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data
		iter_max : int
			maximum number of em steps

		Returns
		-------
		mean : (n_features, ) ndarray
			sample mean of the input data 输入的样本均值
		W : (n_features, n_components) ndarray
			projection matrix 映射矩阵
		var : float
			variance of observation noise 观测噪声的方差
		"""
		initial_list = ["random", "eigen"]
		self.mean = np.mean(X, axis = 0)
		self.I = np.eye(self.n_components)
		if initial not in initial_list:
			print("available initializations are {}". format(initial_list))
		if initial == "random":
			self.W = np.eye(np.size(X, 1), self.n_components)
			self.var = 1.
		elif initial == "eigen":
			self.eigen(X) # 到底是极大似然估计还是特征方法？反正是用来估计参数的，在这里是估计初始参数
		self.alpha = len(self.mean) / np.sum(self.W ** 2, axis = 0).clip(min = 1e-10)
		for i in range(iter_max):
			W = np.copy(self.W)
			stats = self._expectation(X - self.mean)
			self._maximization(X - self.mean, *stats)
			self.alpha = len(self.mean) / np.sum(self.W**2, axis = 0).clip(min = 1e-10) # 这个α的求法不理解？
			if np.allclose(W, self.W):
				break
		self.n_iter = i + 1

	def _maximization(self, X, Ez, Ezz):
		self.W = X.T @ Ez @ np.linalg.inv(np.sum(Ezz, axis = 0) + self.var * np.diag(self.alpha))
		self.var = np.mean(
			np.mean(X**2, axis = -1)
			- 2 * np.mean(Ez @ self.W.T * X, axis = -1)
			+ np.trace((Ezz @ self.W.T @ self.W).T) / len(self.mean))

	def maximize(self, D, Ez, Ezz):
		self.W = D.T.dot(Ez).dot(np.linalg.inv(np.sum(Ezz, axis = 0) + self.var * np.diag(self.alpha)))
		self.var = np.mean(
			np.mean(D**2, axis = -1)
			- 2 * np.mean(Ez.dot(self.W.T) * D, axis = -1)
			+ np.trace(Ezz.dot(self.W.T).dot(self.W).T) / self.ndim)


