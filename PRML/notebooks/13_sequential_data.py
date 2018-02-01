# sequential data
# % matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

from prml.markov_models import CategoricalHMM, GaussianHMM, Kalman, Particle

gaussian_hmm = GaussianHMM(
	initial_proba = np.ones(3) / 3, 
	transition_proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]), 
	means = np.array([[0, 0], [2, 10], [10, 5]]),
	covs = np.asarray([np.eye(2) for _ in range(3)]))

seq = gaussian_hmm.draw(100)

plt.figure(figsize = (10, 10))
plt.scatter(seq[:, 0], seq[:, 1])
plt.plot(seq[:, 0], seq[:, 1], "k", linewidth = 0.1)
for i, p in enumerate(seq):
	plt.annotate(str(i), p)
plt.show()

posterior = gaussian_hmm.fit(seq)

# 这里画的是seq序列数据的图像
plt.figure(figsize = (10, 10))
plt.scatter(seq[:, 0], seq[:, 1], c = np.argmax(posterior, axis = -1)) # 这里和前面是一样的， 但是多了一个posterior， 不同之处就是颜色有改变，之前是一样的颜色，这次三种颜色
plt.plot(seq[:, 0], seq[:, 1], "k", linewidth = 0.1)
for i, p in enumerate(seq):
	plt.annotate(str(i), p)
plt.show()

#################################################################

categorical_hmm = CategoricalHMM(
	initial_proba = np.ones(2) / 2, 
	transition_proba = np.array([[0.95, 0.05], [0.05, 0.95]]), 
	means = np.array([[0.8, 0.2], [0.2, 0.8]]))

seq = categorical_hmm.draw(100)

posterior = categorical_hmm.forward_backward(seq)
hidden = categorical_hmm.viterbi(seq)

plt.plot(posterior[:, 1])
plt.plot(hidden)
for i in range(0, len(seq)):
	plt.annotate(str(seq[i]), (i, seq[i] / 2. + 0.2))
plt.xlim(-1, len(seq))
plt.ylim(-0.1, np.max(seq) + 0.1)
plt.show()

# HMM是先给定模型， 然后根据这个模型生成draw序列， 然后再根据seq序列对这个模型做学习，比如求后验posterior
# 上面主要是讲隐马尔可夫模型， 下面主要是讲滤波
# 滤波的话就是先自己生成数据， 再利用滤波模型进行滤波
##############################################################
###########################################################
seq = np.concatenate(
	(np.arange(50)[:, None] * 0.1 + np.random.normal(size = (50, 1)),
		np.random.normal(loc = 5., size = (50, 1)), # loc此概率分布的均值（对应着整个分布的中心centre）
		5 - 0.1 * np.arange(50)[:, None] + np.random.normal(size = (50, 1))), axis = 0)
seq = np.concatenate((seq, np.gradient(seq, axis = 0)), axis = 1) # Return the gradient of an N-dimensional array. 不知道这里想干嘛， 因为即使没有这一步，所有的结果是一样的，不知道哪里体现梯度了
plt.plot(seq[:, 0])
plt.show()
# 好了，只是暂时生成一个序列数据，不用太纠结

kalman = Kalman(
    transition=np.array([[1, 1], [0, 1]]),
    observation=np.eye(2),
    process_noise=np.eye(2) * 0.01,
    measurement_noise=np.eye(2) * 100,
    init_state_mean=np.zeros(2),
    init_state_cov=np.eye(2) * 100)
mean, var = kalman.filtering(seq)
velocity = mean[:, 1]
mean = mean[:, 0]
std = np.sqrt(var[:, 0, 0]).ravel()

plt.plot(seq[:, 0], label="observation")
plt.plot(mean, label="estimate")
plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color="orange", alpha=0.5, label="error")
plt.legend(bbox_to_anchor=(1, 1))
plt.show()


##############################################################
#########################################################

# 生成一个数字序列， 中间部分是没有的
seq = np.zeros((100, 1)) + np.random.normal(loc = 2., size = (100, 1))
seq[20:80] = np.nan
plt.plot(seq)
plt.show()

##########################################################

kalman = Kalman(
	transition = np.eye(1), 
	observation = np.eye(1),
	process_noise = np.eye(1) * 0.01,
	measurement_noise = np.eye(1),
	init_state_mean = np.zeros(1),
	init_state_cov = np.eye(1) * 100)

mean, var = kalman.filtering(seq)
mean = mean.ravel()
std = np.sqrt(var).ravel()

plt.plot(seq, label = "observation")
plt.plot(mean, label = "estimate")
plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color = "orange", alpha = 0.5, label = "error")
plt.legend(bbox_to_anchor = (1, 1))
plt.show()


# 所谓的滤波，就是得到原始序列的一个估计， 比如说是均值， 这里用了两种：卡尔曼滤波， 粒子滤波。两者都是对上面的序列做的滤波
# 有观测observation， 估计estimate， 和误差error （mean +- std）
#######################################################


def nll(x, particle): # 负对数似然 negative log likelihood , 这个函数也没用到啊
	return np.sum(100. * (x - particle) ** 2, axis = -1)
particle = Particle.filtering(1000, 1, 1)

estimate = particle.filtering(seq)

plt.plot(seq[:, 0], label = "observation")
plt.plot(estimate, label = "estimate")
plt.legend(bbox_to_anchor = (1, 1))
plt.show()
