常见的核函数：
polynomial kernel function

Gaussian kernel function （最常用的RBF）

RBF （radial basis function）径向基函数

输入数据对（x, y），到核函数， 得到对应的核函数值，k(x, y).

与特征有一点区别，后者只输入一个值x， 是对输入做了一个变换； 而前者是把坐标轴做了一个转换，映射到了另一个空间。

http://blog.csdn.net/simplelove17/article/details/44874519
这里贴了PRML的书的图片，讲的是kernel method， 当你遇到想不明白的问题，说明你该看书了。
对于GaussianProcessClassifier这个代码，看看书就明白了。

我自己跟着推导了，过程写在笔记上了。
