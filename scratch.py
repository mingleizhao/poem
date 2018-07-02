import numpy as np
from numba import jit

from poem.matrix.array import empty_arr_3d


@jit(nopython=True)
def tile(a, n):
    k = len(a)
    tmp = np.empty(k * n)
    for i in range(k * n):
        tmp[i] = a[i % k]
    return tmp


@jit(nopython=True)
def repeat(a, n):
    tmp = np.empty(len(a) * n)
    for i in range(len(a)):
        tmp[i*n:(i+1)*n] = a[i]
    return tmp


empty_arr_3d(256)