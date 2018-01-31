# 协方差矩阵的拟合，要借助gram矩阵，因为两者比较相似，可能就差一点小数。

"""
INF表示“无穷大”，是infinite的缩写。
NAN表示“无效数字”，是Not a number的缩写。

inf一般是因为得到的数值，超出浮点数的表示范围（溢出，即阶码部分超过其能表示的最大值）；
而nan一般是因为对浮点数进行了未定义的操作，如对-1开方。
"""

import numpy as np 

class SupportVectorClassifier(object):
	def __init__(self, kernel, C = np.Inf):
		"""
		construct support vector classifier 

		Parameters
		----------
		kernel : Kernel 
			kernel function to compute inner products 用于计算内积的核函数
		C : float
			penalty of misclassification
		"""
		self.kernel = kernel
		self.C = C

	def fit(self, X, t, learning_rate = 0.1, decay_step = 10000, decay_rate = 0.9, min_lr = 1e-5):
		"""
		estimate decision boundary and its support vectors 
		估计决策边界 和 它的支持向量

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data
		t : (sample_size, ) ndarray
			corresponding labels 1 or -1
		learning_rate : float
			update ratio of the lagrange multiplier
			拉格朗日乘子的更新频率
		decay_step : int
			number of iterations till decay 
			迭代多少次之后衰减
		decay_rate : float
			rate of learning rate decay
			学习率的衰减系数
		min_lr : float
			minimum value of learning rate
			学习率的最小值

		Attributes
		----------
		a : (sample_size, ) ndarray
			lagrange multiplier
			拉格朗日乘子
		b : float
			bias parameter
			偏置
		support_vector : (n_vector, n_features) ndarray
			support vectors of the boundary
			支持向量的边界
		"""
		if X.ndim == 1:
			X = X[:, None]
		assert X.ndim == 2
		assert t.ndim == 1
		lr = learning_rate
		t2 = np.sum(np.square(t))
		if self.C == np.Inf:
			a = np.ones(len(t))
		else:
			a = np.zeros(len(t)) + self.C / 10  
			# 感觉就是，如果C罚项太大，那就让拉格朗日乘子a取1； 如果罚项C是给定的值， 那就取拉格朗日乘子为和C相关的值， 又或者我根据下面的内容猜测， 是不是对矩阵和非矩阵的区别？
		Gram = self.kernel(X, X)
		H = t * t[:, None] * Gram
		while True:
			for i in range(decay_step):
				grad = 1 - H @ a
				a += lr * grad # 为什么是加而不是减呢？
				a -= (a @ t) * t / t2 # 这里是怎么回事？
				np.clip(a, 0, self.C, out = a)
			mask = a > 0
			self.X = X[mask]
			self.t = t[mask]
			self.a = a[mask] 
			self.b = np.mean(self.t - np.sum(self.a * self.t * self.kernel(self.X, self.X), axis = -1)) # 求b的式子 p124 ————（2）
			if self.C == np.Inf:
				if np.allclose(self.distance(self.X) * self.t, 1, rtol = 0.01, atol = 0.01):  # Returns True if the two arrays are equal within the given tolerance; False otherwise.
					break # 距离要和1比较
			else:
				if np.all(np.greater_equal(1.01, self.distance(self.X) * self.t)): # 假如我们想要知道矩阵a和矩阵b中所有对应元素是否相等，我们需要使用all方法，假如我们想要知道矩阵a和矩阵b中对应元素是否有一个相等，我们需要使用any方法。
					break # Return the truth value of (x1 >= x2) element-wise.返回大于等于的判定
			if lr < min_lr:
				break
			lr *= decay_rate
# 弄明白决策边界和支持向量，距离，间隔是怎么回事？
"""
support vector支持向量：
在线性可分情况下 ，训练数据集的样本点中与分离超平面距离最近的样本点的实例，称为支持向量。

间隔：
给定一个特定的超平面，我们可以计算出这个超平面与和它最接近的数据点之间的距离。间隔（Margin）就是二倍的这个距离。
一般来说，间隔（Margin）中间是无点区域。这意味着里面不会有任何点。（硬间隔） 软间隔就是中间有噪声的情况。

函数间隔：

几何间隔：
"""
# 因为是求对偶问题，所以，我们要求的参数是a 拉格朗日乘子， 因为w, b都可以用ai来表示， 求出ai后， 自然就求出了w,b的值。
"""
我觉得这个就是P124 的非线性支持向量机学习算法， 有核函数。
实现：
1）就是直接求对（7.95）求导的结果，很简单，gram和H只是式子代换，整个就是迭代的过程，求参数 （这里边有关于a的迭代的两步不是很明白，猜测应该是用的SMO）
（在第二步之前，先选择a>0的部分对应的X，t, a，最优化问题的条件限制）
2）就是算法的第二步求b*

为什么老是有对C 的判断？这里不理解
C > 0 称为惩罚参数， 一般由应用问题决定， C 值大时对误分类的惩罚增大， C值小时对误分类的惩罚减小。
目标函数： 1/2 ||w||^2 + C∑ξ 
最小化目标函数包含两层意思， 使 1/2 ||w||^2尽量小，即间隔尽量大； 同时使误分类点的个数尽量小， C 是调和二者的系数
"""
  # 这里是最优化算法的公式，只不过是相反的值，而且中间少了一个1/2，而且这个函数暂时没用到
	def lagrangian_function(self):
		return (np.sum(self.a) - self.a @ (self.t * self.t[:, None] * self.kernel(self.X, self.X)) @ self.a)

	def predict(self, x):
		"""
		predict labels of the input
		预测输入的标签label

		Parameters
		----------
		x : (sample_size, n_features) ndarray
			input

		Returns
		-------
		label : (sample_size, ) ndarray
			predicted labels
		"""
		y = self.distance(x)
		label = np.sign(y)
		return label

		# 得到的距离要做一个符号函数的变换，才是最终的label

	def distance(self, x):
		"""
		calculate distance from the decision boundary
		计算来自决策边界的距离

		Parameters
		----------
		x : (sample_size, n_features) ndarray
			input

		Returns
		-------
		distance : (sample_size, ) ndarray
			distance from the boundary
		"""
		distance =  np.sum( self.a * self.t * self.kernel(x, self.X), axis = -1) + self.b P124---（3）
		return distance
。
