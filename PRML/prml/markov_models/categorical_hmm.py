import numpy as np
from .hmm import HiddenMarkovModel


class CategoricalHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with categorical emission model HMM的分类排放模型 
    """

    def __init__(self, initial_proba, transition_proba, means):
        """
        construct hidden markov model with categorical emission model
        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray
            probability of initial latent state 初始隐藏状态的概率
        transition_proba : (n_hidden, n_hidden) np.ndarray
            transition probability matrix 转移概率矩阵
            (i, j) component denotes the transition probability from i-th to j-th hidden state 从第i到第j个隐藏状态的转移概率
        means : (n_hidden, ndim) np.ndarray
            mean parameters of categorical distribution 类别分布，分类分布 
        Returns
        -------
        ndim : int
            number of observation categories 观测分类的数目
        n_hidden : int
            number of hidden states 隐藏的状态的数目
        """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0] # 尺寸相同， 转移概率矩阵是方阵
        assert np.allclose(means.sum(axis=1), 1) # mean代表的每一个类别的概率
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means = means

    def draw(self, n=100):
        """
        draw random sequence from this model 从这个模型中获取随机序列，根据给定的状态转移概率
        Parameters
        ----------
        n : int
            length of the random sequence
        Returns
        -------
        seq : (n,) np.ndarray
            generated random sequence
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba) # 根据初始概率， 随机选择隐藏状态
        seq = []
        while len(seq) < n:
            seq.append(np.random.choice(self.ndim, p=self.means[hidden_state])) # 根据隐藏状态所对应的mean概率， 随机选择观测分类数目，加入sequence
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state]) # 根据隐藏状态所对应的转移概率， 随机选择隐藏状态的数目
        return np.asarray(seq)

    def likelihood(self, X):
        return self.means[X] # 就是这个输入X所对应的均值，用来近似代替这个值呗， 或者说，就是要确定mean这个参数

    def maximize(self, seq, p_hidden, p_transition): # 不知道这里做了什么操作，难不成是EM算法？？？这个的参数就是mean。整个过程就是为了求mean。看看Gaussian HMM，有助于理解
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        x = p_hidden[:, None, :] * (np.eye(self.ndim)[seq])[:, :, None]
        self.means = np.sum(x, axis=0) / np.sum(p_hidden, axis=0)
