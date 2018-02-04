# mixture models and em
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
# %matplotlib inline

from prml.clusterings import KMeans
from prml.rv import (
	MultivariateGaussianMixture,
	BernoulliMixture
)

np.random.seed(1111)



# 说到混合模型， 一般情况用kmeans就能解决。 
# 而且既然是混合模型， 那么肯定要事先生成几个正态分布，再组合在一起喽
# k - means clustering
# training data
x1 = np.random.normal(size = (100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size = (100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size = (100, 2))
x3 == np.array([0, 5])
x_train = np.vstack((x1, x2, x3))

x0, x1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
x = np.array([x0, x1]).reshape(2, -1).T

kmeans = KMeans(n_clusters = 3)
kmeans.fit(x_train)
cluster = kmeans.predict(x_train)
plt.scatter(x_train[:, 0], x_train[:, 1], c = cluster)  # 这里的颜色赋值有点问题， 不是颜色名词， 之前也有遇到过，注意
plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s = 200, marker = 'X', lw = 2, c = ['purple', 'cyan', 'yellow'], edgecolor = "white")
plt.contourf(x0, x1, kmeans.predict(x).reshape(100, 100), alpha = 0.1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()

# mixture of gaussians

gmm = MultivariateGaussianMixture(n_components = 3)
gmm.fit(x_train)
p = gmm.classify_proba(x_train)

plt.scatter(x_train[:, 0], x_train[:, 1], c = p)
plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], s=200, marker='X', lw=2, c=['red', 'green', 'blue'], edgecolor="white")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect("equal")
plt.show()

# mixtures of bernoulli distributions

# 说到处理图片，那就拉长成一维向量，这样就没什么特别的了
mnist = fetch_mldata("MNIST original")
x = mnist.data
y = mnist.target
x_train = []
for i in [0, 1, 2, 3, 4]:
    x_train.append(x[np.random.choice(np.where(y == i)[0], 200)])  # 在标签为12345的图片中各选择200张
x_train = np.concatenate(x_train, axis=0)  # 把图片拉长成一维向量
x_train = (x_train > 127).astype(np.float)  # 挑选像素值较大的留下

bmm = BernoulliMixture(n_components=5)
bmm.fit(x_train)

plt.figure(figsize=(20, 5))
for i, mean in enumerate(bmm.mu):
    plt.subplot(1, 5, i + 1)
    plt.imshow(mean.reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.show()
