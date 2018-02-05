"""
# 这一节应该看的还有遗漏，等把RVM for regressor那节的链接看完，还有像极大似然，map这些基础看完，自然就懂了
简单来说，RVM就是基于贝叶斯思想的，涉及概率问题。

相关向量机（Relevance vector machine,简称RVM）是Tipping在2001年在贝叶斯框架的基础上提出的，
它有着与支持向量机（Support vector machine,简称SVM）一样的函数形式，
与SVM一样基于核函数映射将低维空间非线性问题转化为高维空间的线性问题。 

一、RVM与SVM的区别： 
1. SVM 基于结构风险最小化原则构建学习机，RVM基于贝叶斯框架构建学习机 
2. 与SVM相比，RVM不仅获得二值输出，而且获得概率输出 
3. 在核函数的选择上，不受梅西定理的限制，可以构建任意的核函数 
4. 不需对惩罚因子做出设置。在SVM中惩罚因子是平衡经验风险和置信区间的一个常数，实验结果对该数据十分敏感，设置不当会引起过学习等问题。
但是在RVM中参数自动赋值 
5. 与SVM相比，RVM更稀疏，从而测试时间更短，更适用于在线检测。
众所周知，SVM的支持向量的个数随着训练样本的增大成线性增长，
当训练样本很大的时候，显然是不合适的。
虽然RVM的相关向量也随着训练样本的增加而增加，但是增长速度相对SVM却慢了很多。 
6. 学习机有一个很重要的能力是泛化能力，也就是对于没有训练过的样本的测试能力。
文章表明，RVM的泛化能力好于SVM。 
7. 无论是在回归问题上还是分类问题上，RVM的准确率都不亚于SVM。 
8. 但是RVM训练时间长

二、RVM原理步骤 
RVM通过最大化后验概率（MAP）求解相关向量的权重。对于给定的训练样本集{tn,xn}，
类似于SVM , RVM 的模型输出定义为 
y(x;w) = ∑ wi * K(X,Xi) + w0 
其中wi为权重， K(X,Xi)为核函。因此对于, tn=y(xn,w)+εn,
假设噪声εn 服从均值为0 , 方差为σ2 的高斯分布,
则p ( tn | ω,σ2 ) = N ( y ( xi ,ωi ) ,σ2 ) ,
设tn 独立同分布,则整个训练样本的似然函数可以表示出来。

对w 与σ2的求解如果直接使用最大似然法，结果通常使w 中的元素大部分都不是0，从而导致过学习。
在RVM 中我们想要避免这个现像，因此我们为w 加上先决条件：
它们的机率分布是落在0 周围的正态分布: p(wi|αi) = N(wi|0, α?1i ),
于是对w的求解转化为对α的求解，当α趋于无穷大的时候，w趋于0.
"""


import numpy as np

class RelevanceVectorClassifier(object):

    def __init__(self, kernel, alpha=1.):
        """
        construct relevance vector classifier 构建相关向量分类器
        Parameters
        ----------
        kernel : Kernel
            kernel function to compute components of feature vectors 用于计算特征向量的成分的核函数
        alpha : float
            initial precision of prior weight distribution先验权重分布的初始精度
        """
        self.kernel = kernel
        self.alpha = alpha

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _map_estimate(self, X, t, w, n_iter=10):
        for _ in range(n_iter):
            y = self._sigmoid(X @ w)
            g = X.T @ (y - t) + self.alpha * w
            H = (X.T * y * (1 - y)) @ X + np.diag(self.alpha)
            w -= np.linalg.solve(H, g)
        return w, np.linalg.inv(H)

    def fit(self, X, t, iter_max=100):
        """
        maximize evidence with respect ot hyperparameter
        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input
        t : (sample_size,) ndarray
            corresponding target
        iter_max : int
            maximum number of iterations
        Attributes
        ----------
        X : (N, n_features) ndarray
            relevance vector
        t : (N,) ndarray
            corresponding target
        alpha : (N,) ndarray
            hyperparameter for each weight or training sample
        cov : (N, N) ndarray
            covariance matrix of weight
        mean : (N,) ndarray
            mean of each weight
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        assert t.ndim == 1
        Phi = self.kernel(X, X)
        N = len(t)
        self.alpha = np.zeros(N) + self.alpha
        mean = np.zeros(N)
        for _ in range(iter_max):
            param = np.copy(self.alpha)
            mean, cov = self._map_estimate(Phi, t, mean, 10)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            if np.allclose(param, self.alpha):
                break
        mask = self.alpha < 1e8
        self.X = X[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel(self.X, self.X)
        mean = mean[mask]
        self.mean, self.covariance = self._map_estimate(Phi, self.t, mean, 100)

    def predict(self, X):
        """
        predict class label
        Parameters
        ----------
        X : (sample_size, n_features)
            input
        Returns
        -------
        label : (sample_size,) ndarray
            predicted label
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        label = (phi @ self.mean > 0).astype(np.int)
        return label

    def predict_proba(self, X):
        """
        probability of input belonging class one 输入属于类一的概率
        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input
        Returns
        -------
        proba : (sample_size,) ndarray
            probability of predictive distribution p(C1|x) 预测分布的概率， 就是给定数据看属于哪一类
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        mu_a = phi @ self.mean # mean，每一个权重的均值
        var_a = np.sum(phi @ self.covariance * phi, axis=1) # 权重的方差
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
