import numpy as np

import math
import copy
import os

a = np.ones((3, 1))
print(a)
a = np.array([2, 2])
b = np.array([[2, 2], [2, 3]])
print(np.matmul(a, b))

vstep = np.array([1.1, 2.2])
if vstep > 2:
    print(vstep)