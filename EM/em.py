import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = False

# 指定k个高斯分布參数。这里指定k=2。注意2个高斯分布具有同样均方差Sigma，分别为Mu1,Mu2。
# 这里sigma已经确定，只需要估计均值，而且这个是相对简化的版本

def ini_data(Sigma,Mu1,Mu2,k,N):
    global X
    global Mu
    global Expectations
    X = np.zeros((1,N))
    Mu = np.random.random(2)   #要求的参数，先随机初始化一个值
    Expectations = np.zeros((N,k))
    for i in range(0,N):
        if np.random.random(1) > 0.5:   #这个就是和α的选择结合了，也就是事先已经指定了，还有，这是观测数据的生成过程
            X[0,i] = np.random.normal()*Sigma + Mu1 #产生一个指定均值和方差的随机分布矩阵：将randn产生的结果乘以标准差，然后加上期望均值即可。
        else:
            X[0,i] = np.random.normal()*Sigma + Mu2
    if isdebug:
        print ("***********")
        print (u"初始观測数据X：")
        print (X)
# EM算法：步骤1，计算E[zij]
def e_step(Sigma,k,N):
    global Expectations
    global Mu
    global X
    for i in range(0,N):
        Denom = 0
        for j in range(0,k):
            Denom += math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
        for j in range(0,k):
            Numer = math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
            Expectations[i,j] = Numer / Denom
    if isdebug:
        print ("***********")
        print (u"隐藏变量E（Z）：")
        print (Expectations)
# EM算法：步骤2，求最大化E[zij]的參数Mu
def m_step(k,N):
    global Expectations
    global X
    for j in range(0,k):
        Numer = 0
        Denom = 0
        for i in range(0,N):
            Numer += Expectations[i,j]*X[0,i]
            Denom +=Expectations[i,j]
        Mu[j] = Numer / Denom 
# 算法迭代iter_num次。或达到精度Epsilon停止迭代
def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):
    ini_data(Sigma,Mu1,Mu2,k,N)
    print (u"初始<u1,u2>:", Mu)
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)
        e_step(Sigma,k,N)
        m_step(k,N)
        print (i,Mu)
        if sum(abs(Mu-Old_Mu)) < Epsilon:
            break
if __name__ == '__main__':
   run(6,40,20,2,1000,1000,0.0001)
   plt.hist(X[0,:],50)
   plt.show()
   
#转自：http://www.cnblogs.com/slgkaifa/p/6731779.html
