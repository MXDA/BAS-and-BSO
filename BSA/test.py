import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os


x = 2 * np.random.rand(2) - 1
print(x)
print(np.finfo(float).eps)
print(np.linalg.norm([2, 2]))
print(np.sign([-2, 0, 1, 2, -2]))
print(np.ones(3))
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])
print(np.vstack((a, b)))
print(np.vstack((a, [1, 2, 3])))
