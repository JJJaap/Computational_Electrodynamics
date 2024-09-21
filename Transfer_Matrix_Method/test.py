import numpy as np
from scipy.linalg import expm

arr = expm([[1,1],[0,1]])

print(np.exp(arr))