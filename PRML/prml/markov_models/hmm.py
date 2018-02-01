# 我在想这里是不是少写了一个likelihood()函数，或者少导入了一些内容？不然这里涉及的那么多likelihood都是从哪来的？

import numpy as np

class HiddenMarkovModel(object):
	"""
	base class of hidden markov models
	"""

	def __init__(self, initial_proba, transition_proba):
		"""
		construct hidden markov model

		Parameters
		----------
		initial_proba : (n_hidden, ) np.ndarray
			initial probability of each hidden state

		transition_proba : (n_hidden, n_hidden) np.ndarray
			transition probability matrix
			(i, j) component denotes the transition probability from i-th to j-th hidden state

		Attributes 
		----------
		n_hidden : int
			number of hidden state
		"""
		self.n_hidden = initial_proba.size
		self.initial_proba = initial_proba
		self.transition_proba = transition_proba

	def fit(self, seq, iter_max = 100):
		"""
		perform EM algorithm to estimate parameter of emission model and hidden variables
		实施EM算法来估计发行（排放）模型的参数以及隐藏变量 

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
			返回每一个隐变量的后验分布
		"""

		params = np.hstack((self.initial_proba.ravel(), self.transition_proba.ravel()))
		for i in range(iter_max):
			p_hidden, p_transition = self.expect(seq)
			self.maximize(seq, p_hidden, p_transition)
			params_new = np.hstack((self.initial_proba.ravel(), self.transition_proba.ravel()))
			if np.allclose(params, params_new):
				break
			else:
				params = params_new
		return self.forward_backward(seq)

	def expect(self, seq):
		"""
		estimate posterior distributions of hidden states and 
		transition probability between adjacent latent variables
		估计隐藏状态的后验分布 以及 相邻潜在变量的转移概率

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence

		Returns
		-------
		p_hidden : (N, n_hidden) np.ndarray
			posterior distribution of each hidden variable
		p_transition : (N - 1, n_hidden, n_hidden) np.ndarray
			posterior transition probability between adjacent latent variables 相邻潜在变量
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
		p_transition = self.transition_proba * likelihood[1:, None, :] * backward[1:, None, :] * forward[:-1, :, None]
		return p_hidden, p_transition

	def forward_backward(self, seq):
		"""
		就是expect函数的前半部分功能，但是代码几乎一样， 因为求p_transition很简单

		estimate posterior distributions of hidden states

		Parameters
		----------
		seq : (N, ndim) np.ndarray
			observed sequence

		Returns
		-------
		posterior : (N, n_hidden) np.ndarray
			posterior distribution of hidden states
		"""
		likelihood = self.likelihood(seq)

		f = self.initial_proba * likelihood[0]
		constant = [f.sum()]
		forward = [f / f.sum()]
		for like in likelihood[1:]:
			f = forward[-1] @ self.transition_proba * like
			constant.append(f.sum())
			forward.append(f / f.sum())

		backward = [np.ones(self.n_hidden)]
		for like, c in zip(likelihood[-1:0:-1], constant[-1:0:-1]):
			backward.insert(0, self.transition_proba @ (like * backward[0]) / c)

		forward = np.asarray(forward)
		backward = np.asarray(backward)
		posterior = forward * backward
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
		return posterior

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
		seq_hid : (N, ) np.ndarray
			the most probable sequence of hidden variables 输出最优路径对应的隐变量的状态序列
		"""
		nll = -np.log(self.likelihood(seq)) # 负对数似然 negative log likelihood
		cost_total = nll[0]
		from_list = []
		for i in range(1, len(seq)):
			cost_temp = cost_total[:, None] - np.log(self.transition_proba) + nll[i]
			cost_total = np.min(cost_temp, axis = 0)
			index = np.argmin(cost_temp, axis = 0)
			from_list.append(index)
		seq_hid = [np.argmin(cost_total)]
		for source in from_list[::-1]:
			seq_hid.insert(0, source[seq_hid[0]])
		return seq_hid

		# 最优路径不应该是最大概率吗，怎么这里一直在讨论最小？？？
		# 怎么越看越不明白了，这个公式从哪来的？怎么跟书上的算法不一样？
