import random
import numpy as np

# func 和 rv是不一样的， func是我们想要采样的真正的分布， 而rv是我们借助的较简单的可以采样的分布

# 和metropolis方法基本一样，除了接受率公式
def metropolis_hastings(func, rv, n, downsample=1):
    """
    Metropolis Hastings algorith
    Parameters
    ----------
    func : callable
        (un)normalized distribution to be sampled from
    rv : RandomVariable
        proposal distribution
    n : int
        number of samples to draw
    downsample : int
        downsampling factor
    Returns
    -------
    sample : (n, ndim) ndarray
        generated sample
    """
    x = np.zeros((1, rv.ndim))
    sample = []
    for i in range(n * downsample):
        x_new = x + rv.draw()
        accept_proba = func(x_new) * rv.pdf(x - x_new) / (func(x) * rv.pdf(x_new - x))
        if random.random() < accept_proba:
            x = x_new
        if i % downsample == 0:
            sample.append(x[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, rv.ndim), sample.shape
    return sample
