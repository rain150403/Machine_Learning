"""
LDA是一种监督学习的降维技术，也就是说它的数据集的每个样本是有类别输出的。这点和PCA不同。PCA是不考虑样本类别输出的无监督降维技术。
LDA的思想可以用一句话概括，就是“投影后类内方差最小，类间方差最大”。什么意思呢？ 我们要将数据在低维度上进行投影，
投影后希望每一种类别数据的投影点尽可能的接近，而不同类别的数据的类别中心之间的距离尽可能的大。

可能还是有点抽象，我们先看看最简单的情况。假设我们有两类数据 分别为红色和蓝色，如下图所示，这些数据特征是二维的，
我们希望将这些数据投影到一维的一条直线，让每一种类别数据的投影点尽可能的接近，而红色和蓝色数据中心之间的距离尽可能的大。

上图中提供了两种投影方式，哪一种能更好的满足我们的标准呢？从直观上可以看出，右图要比左图的投影效果好，
因为右图的黑色数据和蓝色数据各个较为集中，且类别之间的距离明显。左图则在边界处数据混杂。
以上就是LDA的主要思想了，当然在实际应用中，我们的数据是多个类别的，我们的原始数据一般也是超过二维的，投影后的也一般不是直线，而是一个低维的超平面。
"""


import numpy as np 
from prml.linear.classifier import Classifier
from prml.rv.gaussian import Gaussian

class LinearDiscriminantAnalyzer(Classifier):
	"""
	Linear discriminant analysis model
	"""

	def _fit(self, X, t, clip_min_norm = 1e-10):
		self._check_input(X)
		self._check_target(t)
		self._check_binary(t)
		X0 = X[t == 0]
		X1 = X[t == 1]
		m0 = np.mean(X0, axis = 0)
		m1 = np.mean(X1, axis = 0)
		cov_inclass = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1) # 类内散度矩阵
		self.w = np.linalg.solve(cov_inclass, m1 - m0) # 找特征向量 #solve函数有两个参数a和b。a是一个N*N的二维数组，而b是一个长度为N的一维数组，solve函数找到一个长度为N的一维数组x，使得a和x的矩阵乘积正好等于b，数组x就是多元一次方程组的解。
		self.w /= np.linalg.norm(self.w)clip(min = clip_min_norm) # 正则化
		# 以上是参数拟合，也就是得到w， 
		# 下面是得到输出X · w，变换后的变量Z ， 并对这个输出做高斯拟合
		g0 = Gaussian()
		g0.fit((X0 @ self.w)[:, None])
		g1 = Gaussian()
		g1.fit((X1 @ self.w)[:, None])
		a = g1.var - g0.var
		b = g0.var * g1.mu - g1.var * g0.mu
		c = (g1.var * g0.mu**2 - g0.var * g1.mu**2 - g1.var * g0.var * np.log(g1.var / g0.var))
		self.threshold = (np.sqrt(b**2 - a * c) - b) / a # 这个公式有点像求根公式， 以上几行都是求为了求threshold，这个方法，等以后问问作者吧，自己不研究了

		"""
		norm则表示范数，首先需要注意的是范数是对向量（或者矩阵）的度量，是一个标量（scalar）：
		norm(x, ord=None, axis=None, keepdims=False)
		x表示要度量的向量，ord表示范数的种类
		范数理论的一个小推论告诉我们：ℓ1 ≥ ℓ2 ≥ ℓ∞
		"""

	def transform(self, X):
		"""
		project data
		就是得到映射数据，就是把输入X， 乘以求得的参数w， 映射变量
		Parameters
		----------
		X : (sample_size, n_features) np.ndarray
			input data

		Returns
		-------
		y : (sample_size, 1) np.ndarray
			projected data
		"""
		if not hasattr(self, "w"):
			raise AttributeError("perform fit method to estimate linear projection")
		return X @ self.w

	def _classify(self, X):
		return (X @ self.w > self.threshold).astype(np.int) # 判别边界系数

"""
输入：数据集D={(x1,y1),(x2,y2),...,((xm,ym))}D={(x1,y1),(x2,y2),...,((xm,ym))},其中任意样本xixi为n维向量，yi∈{C1,C2,...,Ck}yi∈{C1,C2,...,Ck}，降维到的维度d。

输出：降维后的样本集$D′$

　　　　1) 计算类内散度矩阵SwSw

　　　　2) 计算类间散度矩阵SbSb

　　　　3) 计算矩阵S−1wSbSw−1Sb

　　　　4）计算S−1wSbSw−1Sb的最大的d个特征值和对应的d个特征向量(w1,w2,...wd)(w1,w2,...wd),得到投影矩阵[Math Processing Error]W

　　　　5) 对样本集中的每一个样本特征xixi,转化为新的样本zi=WTxizi=WTxi
　　　　6) 得到输出样本集D′={(z1,y1),(z2,y2),...,((zm,ym))}D′={(z1,y1),(z2,y2),...,((zm,ym))}
 
以上就是使用LDA进行降维的算法流程。实际上LDA除了可以用于降维以外，还可以用于分类。一个常见的LDA分类基本思想是假设各个类别的样本数据符合高斯分布，
这样利用LDA进行投影后，可以利用极大似然估计计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。当一个新的样本到来后，我们可以将它投影，
然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。

"""
