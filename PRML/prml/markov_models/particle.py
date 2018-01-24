# 粒子叫作估计器estimator。估计过去叫平滑smoothing，估计未来叫预测prediction，估计当前值才叫滤波filtering。 说的太好了

import numpy as np 
from scipy.misc import logsumexp
from .state_space_model import StateSpaceModel

class Particle(StateSpaceModel):
	"""
	a class to perform particle filtering or smoothing
	"""

	def __init__(self, n_particles):
		"""
		construct state space model to perform particle filtering or smoothing

		Parameters
		----------
		n_particles : int
			number of particles 粒子数量
		sigma : int or float
			standard deviation of gaussian transition 高斯转换的标准差
		ndim : int
			dimensionality
		nll : callable
			negative log likelihood
		"""

		self.n_particles = n_particles
		self.sigma = sigma
		self.ndim = ndim
		if nll is None:
			def nll(obs, particle):
				return np.sum((obs - particle)**2, axis = -1)
		self.nll = nll

	def likelihood(self, X, particle):
		logit = -self.nll(X, particle)
		logit -= logsumexp(logit) # Compute the log of the sum of exponentials of input elements.
		weight = np.exp(logit)
		assert np.allclose(weight.sum(), 1.), weight.sum()
		assert weight.shape == (len(particle), ), weight.shape
		return weight

	def resample(self, particle, weight):
		index = np.random.choice(len(particle), size = len(particle), p = weight)
		return particle[index]

	def filtering(self, seq):
		"""
		particle filtering

		1. prediction
			p(z_n+1|x_1:n) = \int p(z_n+1|z_n) p(z_n|x_1:n)dz_n
		2. filtering
			p(z_n+1|x_1:n+1) \propto p(x_n+1|z_n+1)p(z_n+1|x_1:n)

		Parameters
		----------
		seq : (N, ndim_observe) np.ndarray
			observed sequence

		Returns
		-------
		output : type
			explanation of the output 输出的说明
		"""
		self.position = []
		position = np.random.normal(size = (self.n_particles, self.ndim))
		for obs in seq:
			delta = np.random.normal(scale = self.sigma, size = (self.n_particles, self.ndim))
			position = position + delta # 这里是个小技巧吗？
			weight = self.likelihood(obs, position)
			position = self.resample(position, weight)
			self.position.append(position)
		self.position = np.asarray(self.position)
		return self.position.mean(axis = 1)
	def smoothing(self):
		pass


"""
粒子滤波的作用就是filtering过滤或者smoothing平滑，这也是一种state space model状态空间模型
这里涉及的函数有:
__init__()设置参数，属性，粒子数量， 标准差， 维数， 负对数似然
likelihood（）似然函数，公式倒是很好找，但是就是不知道怎么就是结果了，weight, 其实就是求相似度，不知道怎么就弄得这么麻烦，待思考
resample（）重采样 随机采样，按weight权重采样
filtering（）滤波

从正态分布中生成position
对于序列中每一个观测， 利用观测obs和新的position的似然估计得到weight； 按weight在新position中重采样；
最后对整体取均值

smoothing（）平滑


粒子滤波算法源于蒙特卡洛思想，即以某事件出现的频率来指代该事件的概率。通俗的讲，粒子滤波也是能用已知的一些数据预测未来的数据。我们知道，科尔曼滤波限制
噪声时服从高斯分布的，但是粒子滤波可以不局限于高斯噪声，原理上粒子滤波可以驾驭所有的非线性、非高斯系统。
一个比喻：
某年月，警方（跟踪程序）要在某个城市的茫茫人海（采样空间）中跟踪寻找一个罪犯（目标），警方采用了粒子滤波的方法。
1. 初始化：
警方找来了一批警犬（粒子），并且让每个警犬预先都闻了罪犯留下来的衣服的味道（为每个粒子初始化状态向量S0），然后将警犬均匀布置到城市的各个区（均匀分布是
初始化粒子的一种方法，另外还有诸如高斯分布，即：将警犬以罪犯留衣服的那个区为中心来扩展分布开来）。
2. 搜索：
每个警犬都闻一闻自己位置的人的味道（粒子状态向量Si），并且确定这个味道跟预先闻过的味道的相似度（计算特征向量的相似性），这个相似度的计算最简单的方法
就是计算一个欧式距离（每个粒子i对应一个相似度Di），然后做归一化（即：保证所有粒子的相似度之和为1）。
3. 决策：
总部根据警犬们发来的味道相似度确定罪犯出现的位置（概率上最大的目标）：最简单的决策方法为哪个味道的相似度最高，那个警犬处的人就是目标。
4. 重采样：
总部根据上一次的决策结果，重新布置下一轮警犬分布（重采样过程）。最简单的方法为：把相似度比较小的地区的警犬抽调到相似度高的地区。
上述，2,3,4过程重复进行，就完成了粒子滤波跟踪算法的全过程。

 粒子滤波的核心思想是随机采样+重要性重采样。既然不知道目标在哪里，那我就随机的放狗（随机采样）。放完狗后，根据特征相似度计算每个地区人和罪犯的相似度，
 然后在重要的地方再多放狗，不重要的地方就少放狗（重要性采样）。
 """

