# 如果你遇到不懂的了，那就往下看看，那就多看看类似的，类比一下，说不定就明白了

import numpy as np
from prml.random import Gaussian
from .hmm import HiddenMarkovModel


class GaussianHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with Gaussian emission model
    """

    def __init__(self, initial_proba, transition_proba, means, covs):
        """
        construct hidden markov model with Gaussian emission model
        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray or None
            probability of initial states
        transition_proba : (n_hidden, n_hidden) np.ndarray or None
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        means : (n_hidden, ndim) np.ndarray
            mean of each gaussian component 每一部分的系数（概率）？或者每一高斯部分的均值
        covs : (n_hidden, ndim, ndim) np.ndarray
            covariance matrix of each gaussian component 每一高斯部分的方差
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
        draw random sequence from this model
        Parameters
        ----------
        n : int
            length of the random sequence
        Returns
        -------
        seq : (n, ndim) np.ndarray
            generated random sequence
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.extend(self.gaussians[hidden_state].draw())
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def likelihood(self, X):
        diff = X[:, None, :] - self.means
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)

    def maximize(self, seq, p_hidden, p_transition):
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        Nk = np.sum(p_hidden, axis=0)
        self.means = (seq.T @ p_hidden / Nk).T
        diffs = seq[:, None, :] - self.means
        self.covs = np.einsum('nki,nkj->kij', diffs, diffs * p_hidden[:, :, None]) / Nk[:, None, None]

