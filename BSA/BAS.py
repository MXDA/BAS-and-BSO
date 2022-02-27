import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os


def FitFunc(xx):  # 测试函数
    x1 = xx[0]
    x2 = xx[1]
    frac1 = 1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))
    frac2 = 0.5 * (x1 ** 2 + x2 ** 2) + 2
    y = -frac1 / frac2
    return y


def Bounds(x, lb, ub):
    for i in range(np.size(x)):
        if x[i] < lb[i]:
            x[i] = lb[i]
        if x[i] > ub[i]:
            x[i] = ub[i]
    return x


#############参数设置####################
print("******************** BAS Algorithm 天牛须********************")

k = 2  # 维数
lb = -5.12 * np.ones(k)  # 下界
ub = 5.12 * np.ones(k)  # 上界
n = 100  # 迭代次数

# %注意：初始步长和步长衰减系数的设置对于算法的寻优速度和寻优精度有影响
d = 2  # 初始化天牛触须长度
eta = 0.95  # 触须长度衰减系数  维数越大，eta设置要越接近1
c = 5  # step = c*d; 步长系数

############天牛初始化######################
x0 = np.random.rand(k)  # 天牛位置初始化，rand只能生成正数，rands生成的可正可负
'''
for i in range(np.size(lb)):
    x0[i] = x0[i] * (ub[i] - lb[i])
x0 = lb + x0
'''
x = x0
xbest = x0  # 记录最优位置
fbest = FitFunc(x0)  # 记录最优适应度值
fbest_store = fbest  # 存储历史最优值
x_store = np.array([0, x, fbest])  # 保存了迭代次数、当前最优位置和相应的适应度值
print(x_store)
#######开始n次迭代#####################
for i in range(1, n + 1):
    step = c * d  # 步长计算
    dir = 2 * np.random.rand(k) - 1  # 天牛随机朝向
    dir = dir / (np.finfo(float).eps + np.linalg.norm(dir))  # eps：很小的一个数。 2-范数：sqrt(│x1│^2+│x2│^2+......)
    xleft = x + dir * d / 2  # 天牛左触角探测位置
    fleft = FitFunc(xleft)
    xright = x - dir * d / 2  # 天牛右触角探测位置
    fright = FitFunc(xright)

    x = x - step * dir * np.sign(fleft - fright)  # 天牛位置更新
    x = Bounds(x, lb, ub)
    f = FitFunc(x)

    if f < fbest:
        xbest = x
        fbest = f
        BAS_iteration = i + 1  # 记录最最早在哪次迭代时到达最优值
    x_store = np.vstack((x_store, [i, x, f]))
    fbest_store = np.append(fbest_store, fbest)
    print('第 ', i, '次迭代', 'x=', xbest, 'y=', fbest)

    d = d * eta  # 触须长度衰减

##############打印最优值######################
print('fbest = ', fbest)
x_label = x_store[:, 0]
y_label = fbest_store
plt.plot(x_label, y_label, markeredgecolor="black", linewidth=2)
plt.xlabel("iteration")
plt.ylabel("minimum vaLue")
plt.show()