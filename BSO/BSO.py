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


# 粒子参数设置

Dim = 2  # 维数
lb = -5.12
ub = 5.12
Lb = lb * np.ones(Dim)  # 下界
Ub = ub * np.ones(Dim)  # 上界

SwarmSize = 30  # 粒子群种群数
MaxIter = 100  # 最大迭代次数

w = 0.6  # 惯性因子 0.6
c1 = 2  # 加速常数 学习因子
c2 = 2  # 加速常数 学习因子

Vmax = 1
Vmin = -1

# 天牛参数设置
d = 3  # 初始触须长度
eta = 0.98  # 触须衰减因子
c = 5  # step = c*d;
c3 = 0.1  # 学习因子
langda = 0.9  # 权重

# 粒子群初始化
Swarm = np.random.rand(SwarmSize, Dim) * (Ub - Lb) + np.ones((SwarmSize, Dim)) * lb  # 初始化粒子群
VStep = np.random.rand(SwarmSize, Dim) * (Vmax - Vmin) + Vmin  # 初始化速度

fSwarm = np.zeros(SwarmSize)
for i in range(SwarmSize):
    fSwarm[i] = FitFunc(Swarm[i, :])  # 初始化之后求出每一个粒子群的适应值
# 个体极值和群体极值
bestf = np.min(fSwarm)
bestindex = np.argmin(fSwarm)
fzbest = bestf  # 全局最佳适应值
zbest = Swarm[bestindex, :]  # 全局最佳
gbest = Swarm  # 个体最佳
fgbest = fSwarm  # 个体最佳适应值

iter = 0
FF = np.zeros(MaxIter)
while iter < MaxIter:  # 当迭代次数还没达到最大且全局最优适应值大于最小适应值时，循环继续
    for j in range(SwarmSize):
        # 天牛左右触须探测位置
        step = c * d
        Xr = Swarm[j, :] + (VStep[j, :] * d / 2)
        Xl = Swarm[j, :] - (VStep[j, :] * d / 2)
        gema = step * VStep[j, :] * np.sign(FitFunc(Xr) - FitFunc(Xl))
        fct1 = w * VStep[j, :] + c1 * np.random.rand() * (gbest[j, :] - Swarm[j, :]) + \
               c2 * np.random.rand() * (zbest - Swarm[j, :])
        fct2 = c3 * (2 * np.random.rand() - 1) * gema
        VStep[j, :] = langda * fct1 + (1 - langda) * fct2
        # 原始速度更新
        for k in range(Dim):
            if VStep[j, k] > Vmax:
                VStep[j, k] = Vmax
            if VStep[j, k] < Vmin:
                VStep[j, k] = Vmin
        # 位置更新
        Swarm[j, :] = Swarm[j, :] + VStep[j, :]
        for k in range(Dim):
            if Swarm[j, k] > Ub[k]:
                Swarm[j, k] = Ub[k]
            if Swarm[j, k] < Lb[k]:
                Swarm[j, k] = Lb[k]
        # 适应值
        fSwarm[j] = FitFunc(Swarm[j, :])
        # 个体最优更新
        if fSwarm[j] < fgbest[j]:
            gbest[j] = Swarm[j]
            fgbest[j] = fSwarm[j]

        # 群体最优更新
        if fSwarm[j] < fzbest:
            zbest = Swarm[j, :]
            fzbest = fSwarm[j]
            BAS_PSO_iteration = iter  # 记录最最早在哪次迭代时到达群体最优值
    FF[iter] = fzbest
    iter = iter + 1  # 迭代次数更新

    d = d * eta  # 更新天牛触角长度
    print('第 ', iter, '次迭代', 'x=', zbest, 'y=', fzbest)

print('fzbest = ', fzbest)



plt.plot(FF, linewidth = 2)
plt.xlabel("Number of Iterarions")
plt.ylabel("Function Values")
plt.show()