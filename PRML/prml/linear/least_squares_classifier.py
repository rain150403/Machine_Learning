
import numpy as np 
from prml.linear.classifier import Classifier

class LeastSquaresClassifier(Classifier):
	"""
	least squares classifier model
	y = argmax_k X @ W
	"""
	def __init__(self, W = None):
		self.W = W

	def _fit(self, X, t):
		self._check_input(X)
		self._check_target(t)
		T = np.eye(int(np.max(t)) + 1)[t]
		self.W = np.linalg.pinv(X) @ T # 伪逆矩阵 y = w * x + b = W * X , 所以W = X^-1 * y

		"""
		>>> c = [1, 0, 0, 1, 1]
		>>> d = np.max(c)
		>>> d
		1
		>>> e = np.eye(2)
		>>> e
		array([[ 1.,  0.],
       			[ 0.,  1.]])
		>>> e[c]
		array([[ 0.,  1.],
       			[ 1.,  0.],
       			[ 1.,  0.],
      			[ 0.,  1.],
       			[ 0.,  1.]])
		>>>
		我觉得这一句就是把本是一维向量的target转换成one-hot向量的堆叠


		对于方阵A，如果为非奇异方阵，则存在逆矩阵inv(A)
对于奇异矩阵或者非方阵，并不存在逆矩阵，但可以使用pinv(A)求其伪逆

inv：

inv(A)*B
实际上可以写成A\B
B*inv(A)
实际上可以写成B/A
这样比求逆之后带入精度要高
A\B=pinv(A)*B 
A/B=A*pinv(B)

pinv：

X=pinv(A),X=pinv(A,tol),其中tol为误差
pinv是求广义逆
先搞清楚什么是伪逆。
对于方阵A，若有方阵B，使得：A·B=B·A=I，则称B为A的逆矩阵。
如果矩阵A不是一个方阵，或者A是一个非满秩的方阵时，矩阵A没有逆矩阵，但可以找到一个与A的转置矩阵A'同型的矩阵B，使得：
     A·B·A=A        
      B·A·B=B
此时称矩阵B为矩阵A的伪逆，也称为广义逆矩阵。因此伪逆阵与原阵相乘不一定是单位阵。
当A可逆时，B就是A的逆矩阵，否则就是广义逆。
满足上面关系的A,B矩阵，有很多和逆矩阵相似的性质。

如果A为非奇异矩阵的话，虽然计算结果相同，但是pinv会消耗大量的计算时间。
在其他情况下，pinv具有inv的部分特性，但是不完全相同。

		"""

	def _classify(self, X):
		return np.argmax(X @ self.W, axis = -1) # Returns the indices of the maximum values along an axis.

# 这里的最小二乘方法怎么这样简单
