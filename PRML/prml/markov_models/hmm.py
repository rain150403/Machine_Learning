# 我在想这里是不是少写了一个likelihood()函数，或者少导入了一些内容？不然这里涉及的那么多likelihood都是从哪来的？
# 而且maximize函数也没有写，这个函数应该是输入seq, p_hidden, p_transition，返回initial_proba, transition_proba

import numpy as np

class HiddenMarkovModel(object):
	"""
	base class of hidden markov models HMM的基础类
	"""

	def __init__(self, initial_proba, transition_proba):
		"""
		construct hidden markov model 构建隐马尔科夫模型

		Parameters
		----------
		initial_proba : (n_hidden, ) np.ndarray
			initial probability of each hidden state 每一个隐藏状态的初始概率

		transition_proba : (n_hidden, n_hidden) np.ndarray
			transition probability matrix 转移概率矩阵
			(i, j) component denotes the transition probability from i-th to j-th hidden state
			(i, j)成分表示，从第i个到第j个隐藏状态的转移概率， 也就是说，它只是隐藏状态之间的转移

		Attributes 
		----------
		n_hidden : int
			number of hidden state 隐状态的数量
		"""
		self.n_hidden = initial_proba.size
		self.initial_proba = initial_proba
		self.transition_proba = transition_proba

	def fit(self, seq, iter_max = 100):
		"""
		perform EM algorithm to estimate parameter of emission model and hidden variables
		实施EM算法来估计发行（排放）模型的参数以及隐藏变量，
		所谓拟合，无非就是估计模型的参数，隐变量等， 所使用的方法就是EM算法

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence (观测序列)
		iter_max : int
			maximum number of EM steps （最大迭代次数）

		Returns
		-------
		posterior : (N, n_hidden) np.ndarray
			posterior distribution of each latent variable 
			返回每一个隐变量的后验分布， 
			至于后验分布，就是该变量在某些参数的条件下，生成的 比如p(h|x, alpha), h 就是隐变量 x是观测数据 alpha是参数
		"""

		params = np.hstack((self.initial_proba.ravel(), self.transition_proba.ravel())) # 参数在这里指的是初始概率和转移概率
		for i in range(iter_max):
			p_hidden, p_transition = self.expect(seq) # seq是观测序列，p_hidden是每一个隐变量的后验分布，p_transition是相邻潜在变量的转移概率的后验
			self.maximize(seq, p_hidden, p_transition)   # EM算法要传入seq序列数据才能运作嘛
			params_new = np.hstack((self.initial_proba.ravel(), self.transition_proba.ravel()))
			if np.allclose(params, params_new):
				break     # 也是用迭代加EM算法来求参数， 最后就是判断第i次与第j次迭代的结果是否相等
			else:
				params = params_new
		return self.forward_backward(seq)    # 为什么求了半天参数， 最后返回的只是seq的forward_backward,与函数内容完全无关 这样可以得到观测的后验概率，可能正是我们想要的

	def expect(self, seq):
		"""
		estimate posterior distributions of hidden states and 
		transition probability between adjacent latent variables
		估计隐藏状态的后验分布 以及 相邻潜在变量的转移概率 
		既然我们是要求隐变量和参数，那这里求隐变量的后验分布作为代替， 而转移概率就是我们的参数喽 
		
		和forward_backward函数一样

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence 观测序列

		Returns
		-------
		p_hidden : (N, n_hidden) np.ndarray
			posterior distribution of each hidden variable 每一个隐变量的后验分布
		p_transition : (N - 1, n_hidden, n_hidden) np.ndarray
			posterior transition probability between adjacent latent variables 相邻潜在变量之间的转移概率的后验
		"""
		likelihood = self.likelihood(seq)

		f = self.initial_proba * likelihood[0]
		constant = [f.sum()]
		forward = [f / f.sum()]
		for like in likelihood[1:]:
			f = forward[-1] @ self.transition_proba * like
			constant.append(f.sum())
			forward.append(f / f.sum())
		forward = np.asarray(forward)
		constant = np.asarray(constant)

		backward = [np.ones(self.n_hidden)]
		for like, c in zip(likelihood[-1:0:-1], constant[-1:0:-1]) :
			backward.insert(0, self.transition_proba @ (like * backward[0]) / c)
		backward = np.asarray(backward)

		p_hidden = forward * backward
		p_transition = self.transition_proba * likelihood[1:, None, :] * backward[1:, None, :] * forward[:-1, :, None] # 是ξ(i, j)的式子吗？P179
		return p_hidden, p_transition
	
	def forward_backward(self, seq):
		"""
		就是expect函数的前半部分功能，但是代码几乎一样， 因为求p_transition很简单，
		最后还有一个把前向概率求和的过程，而前向概率就是观测序列在给定模型下的概率

		estimate posterior distributions of hidden states 估计隐状态的后验分布 
		正常forward_backward算法是用来计算观测序列的后验概率

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence

		Returns
		-------
		posterior : (N, n_hidden) np.ndarray
			posterior distribution of hidden states 可书上说的是观测的后验呀
		"""
		likelihood = self.likelihood(seq) # 如此说来，依然是观测数据喽

		f = self.initial_proba * likelihood[0] # 初始前向概率 = 初始状态概率 * 观测概率 α1(i) = πi*bi(o1) 。初始化前向概率， 是初始时刻的状态i1 = qi和观测o1的联合概率。
		constant = [f.sum()]
		forward = [f / f.sum()] # 这里也是归一化吧？
		for like in likelihood[1:]:
			f = forward[-1] @ self.transition_proba * like # 这个就是P176--（2） 就是前向概率和 * 转移概率 * 观测， 也就是递推公式
			constant.append(f.sum())
			forward.append(f / f.sum())
########以上就是前向算法， P176
		backward = [np.ones(self.n_hidden)]
		for like, c in zip(likelihood[-1:0:-1], constant[-1:0:-1]):
			backward.insert(0, self.transition_proba @ (like * backward[0]) / c)  # 这里是P178--（2）就是后向算法的第2步 

		forward = np.asarray(forward)
		backward = np.asarray(backward)
		posterior = forward * backward  # 利用前向概率和后向概率的定义可以将观测序列概率P(O|λ)统一写成
		return posterior 

	def filtering(self, seq):
		"""
		bayesian filtering
		贝叶斯过滤器通过使用贝叶斯逻辑， 根据概率把物品分类

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence

		Returns
		-------
		posterior : (N, n_hidden) np.ndarray
			posterior distributions of each latent variables
		"""
		likelihood = self.likelihood(seq)
		p = self.initial_proba * likelihood[0]
		posterior = [p / np.sum(p)]
		for like in likelihood[1:]:
			p = posterior[-1] @ self.transition_proba * likelihood
			posterior.append(p / np.sum(p))
		posterior = np.asarray(posterior)
		return posterior # 就是前向计算 可以得到概率，后验概率，这个概率就可以用于分类

	def viterbi(self, seq):
		"""
		viterbi algorithm (a.k.a max-sum algorithm) also known as 最大和算法 https://wenku.baidu.com/view/dc76163443323968011c9283.html
		维特比算法， 用动态规划解隐马尔科夫模型预测问题，即用动态规划求概率最大路径（最优路径）， 这时一条路径对应着一个状态序列。

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence 输入观测序列

		Returns
		-------
		seq_hid : (N, ) np.ndarray # 输出隐变量最可能的序列
			the most probable sequence of hidden variables 输出最优路径对应的隐变量的状态序列
		"""
		nll = -np.log(self.likelihood(seq)) # 负对数似然 negative log likelihood
		cost_total = nll[0]
		from_list = []
		for i in range(1, len(seq)):
			cost_temp = cost_total[:, None] - np.log(self.transition_proba) + nll[i] # 这里就是把所有的对数似然加在一起算总的损失
			cost_total = np.min(cost_temp, axis = 0)
			index = np.argmin(cost_temp, axis = 0)
			from_list.append(index)
		seq_hid = [np.argmin(cost_total)]
		for source in from_list[::-1]:
			seq_hid.insert(0, source[seq_hid[0]])
		return seq_hid

		# 最优路径不应该是最大概率吗，怎么这里一直在讨论最小？？？ 损失最小，结果不就最优嘛
		# 怎么越看越不明白了，这个公式从哪来的？怎么跟书上的算法不一样？暂时就这样吧，以后再研究
