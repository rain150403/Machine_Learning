# 一般的模型，就是两个函数 拟合fit（求出模型参数）， 预测predict（根据X，利用求得的模型， 计算y）
# 一般是模型结构已经有了，但是参数没定，那就要用大量数据去拟合模型，也就是得到参数的确定值，这时模型就完全确定了，那就可以应用了，输入X，就能得到对应的y值。

import numpy as np 
from scipy.spatial.distance import cdist # Computes distance between each pair of the two collections of inputs.

class KMeans(object):
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters

	def fit(self, X, iter_max = 100):
		"""
		perform k-means algorithm
		输入数据和最大迭代次数，输出聚类中心

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data
		iter_max : int
			maximum number of iterations

		Returns
		-------
		centers : (n_clusters, n_features) ndarray
			center of each cluster
		"""
		I = np.eye(self.n_clusters) # 有几个聚类中心，就产生几维单位矩阵
		centers = X[np.random.choice(len(X), self.n_clusters, replace = False)] # 随机选择n_cluster个样本做聚类中心
		for _ in range(iter_max): # 开始迭代修改聚类中心
			prev_centers = np.copy(centers)
			D = cdist(X, centers) # 计算每一个样本到每一个聚类中心的距离
			cluster_index = np.argmin(D, axis = 1) # 在距离矩阵中选择每一行的最小值索引
			cluster_index = I[cluster_index] # 根据上面得到的索引选择单位矩阵的对应位置
			centers = np.sum(X[:, None, :] * cluster_index[:, :, None], axis = 0) / np.sum(cluster_index, axis = 0)[:, None] # 得到新的中心，根据求质心的公式
			if np.allclose(prev_centers, centers): # 比较旧的中心和新的中心，如果聚类中心基本不再变化， 就停止迭代
				break
		self.centers = centers

	def predict(self, X):
		"""
		calculate closest cluster center index
		计算离它最近的聚类中心的索引，也就是告诉我，它离哪个类别最近，就属于那个类

		Parameters
		----------
		X : (sample_size, n_features) ndarray
			input data

		Returns
		-------
		index : (sample_size, ) ndarray
			indicates which cluster they belong  看看它属于哪个类
		"""

		# 计算输入与各个聚类中心的距离， 选择最小的那个
		D = cdist(X, self.centers)
		return np.argmin(D, axis = 1)
