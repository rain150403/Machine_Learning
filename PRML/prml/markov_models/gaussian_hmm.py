# 如果你遇到不懂的了，那就往下看看，那就多看看类似的，类比一下，说不定就明白了

import numpy as np
from prml.random import Gaussian
from .hmm import HiddenMarkovModel


class GaussianHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with Gaussian emission model 高斯模型和隐马尔可夫模型
    """

    def __init__(self, initial_proba, transition_proba, means, covs):
        """
        construct hidden markov model with Gaussian emission model 用高斯模型构建HMM
        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray or None
            probability of initial states 初始状态概率
        transition_proba : (n_hidden, n_hidden) np.ndarray or None
            transition probability matrix 转移概率矩阵， 是隐状态之间的转换
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        means : (n_hidden, ndim) np.ndarray
            mean of each gaussian component 每一部分的系数（概率）？或者每一高斯部分的均值
        covs : (n_hidden, ndim, ndim) np.ndarray
            covariance matrix of each gaussian component 每一高斯部分的方差矩阵
        Attributes
        ----------
        ndim : int
            dimensionality of observation space 观测空间的维数
        n_hidden : int
            number of hidden states 隐藏状态的数量
        """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0] == covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means = means
        self.covs = covs
        self.precisions = np.linalg.inv(self.covs) 
        self.gaussians = [Gaussian(m, cov) for m, cov in zip(means, covs)]

    def draw(self, n=100):
        """
        draw random sequence from this model 从这个模型中获得draw随机序列
        Parameters
        ----------
        n : int
            length of the random sequence  随机序列的长度
        Returns
        -------
        seq : (n, ndim) np.ndarray
            generated random sequence  生成的随机序列
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)  # 根据初始概率，以及要求的隐变量的数量， 随机产生隐藏状态
        seq = []
        while len(seq) < n:
            seq.extend(self.gaussians[hidden_state].draw()) # 根据隐状态对应的高斯分布， 产生seq序列 加入 list中
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state]) # 根据隐变量的数量，以及隐状态对应的转移概率， 随机选择新的隐状态
        return np.asarray(seq)

    def likelihood(self, X):
        diff = X[:, None, :] - self.means
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)  # 这个似然函数只是返回了高斯公式
# 似然，就是求似然函数呗， 这里就是高斯公式喽
    
    """
     p_hidden : posterior distribution of each hidden variable 每一个隐变量的后验分布
     p_transition :posterior transition probability between adjacent latent variables 相邻潜在变量之间的后验转移概率
    """
    def maximize(self, seq, p_hidden, p_transition):
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])  # 占总体的百分比就是初始概率
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2)) # 同上
        Nk = np.sum(p_hidden, axis=0)
        self.means = (seq.T @ p_hidden / Nk).T
        diffs = seq[:, None, :] - self.means
        self.covs = np.einsum('nki,nkj->kij', diffs, diffs * p_hidden[:, :, None]) / Nk[:, None, None] # 求方差的公式
# 最大化，就是求参数喽，这里就是求均值和方差 ，求参当然要知道参数的通常求法及公式喽， 对应的模型，方法，概率不同，所求的参数自然也不同，虽然使用的都是EM算法，但是也是不同问题不同分析
    
   
"""
np.einsum : 任意维度张量之间的广义收缩。

tf.einsum

einsum(
    equation,
    *inputs
)

这个函数返回一个张量，其元素其元素是由等式定义的，这是由爱因斯坦求和公式所启发的速写形式定义的。作为示例，考虑将两个矩阵 A 和 B 相乘以形成矩阵C。
C的元素由下式给出：
C[i,k] = sum_j A[i,j] * B[j,k]
相应的等式是：
ij,jk->ik

一般来说, 方程是从较熟悉的元素方程得到：
删除变量名称、括号和逗号；
用 "*" 替换 "，"；
删除总和标志；
将输出移到右侧，并将 "=" 替换为 "->>"。

许多常见操作可以用这种方式来表示。例如:
# Matrix multiplication
>>> einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]

# Dot product
>>> einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

# Outer product
>>> einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]

# Transpose
>>> einsum('ij->ji', m)  # output[j,i] = m[i,j]

# Batch matrix multiplication
>>> einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]

此函数的行为类似于 numpy.einsum，但不支持：
椭圆（下标像：ij...,jk...->ik...）
一个轴在单个输入上出现多次的下标（例如 ijj、k->> ik）。
在多个输入之间求和的下标（例如 ij、ij、jk->> ik）。

ARGS：
equation：一种描述收缩的 str，与 numpy.einsum 格式相同。
* inputs：合同的输入（每个张量），其形状应与 equation 一致。

返回：
返回收缩的张量，形状由 equation 决定。

注意：
ValueError：如果 equation 格式不正确，equation 隐含的输入数与 len(inputs) 不匹配，一个轴出现在输出下标中，但不显示在任何输入中，
输入的维数与其下标中的索引数不同，或者输入形状沿特定轴线不一致。 


"""
