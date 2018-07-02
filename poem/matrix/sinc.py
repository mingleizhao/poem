# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


from math import sin
from math import sqrt
import numpy as np
from numba import jit
from poem.matrix.array import find_indices_2d
from poem.matrix.array import find_indices_3d
from poem.matrix.array import indices_2d
from poem.matrix.array import indices_3d


def v_sinc_2d(original_size, pad, axes, mode='half'):
    padded_size = original_size * pad
    mat = indices_2d(original_size, axes=axes, mode=mode)
    r_mat = np.linalg.norm(mat, axis=1) / padded_size
    sinc_mat = np.sin(r_mat * np.pi) / (r_mat * np.pi)

    # sinc(0) = 1
    sinc_mat[find_indices_2d(0, 0, original_size, axes, mode)] = 1.0

    if mode == 'half':
        return sinc_mat.reshape((original_size, original_size // 2 + 1))
    elif mode == 'full':
        return sinc_mat.reshape((original_size, original_size))
    else:
        raise ValueError('sinc_2d: mode not defined!')


def v_sinc_3d(original_size, pad, axes, mode='half'):
    padded_size = original_size * pad
    mat = indices_3d(original_size, axes=axes, mode=mode)
    r_mat = np.linalg.norm(mat, axis=1) / padded_size
    sinc_mat = np.sin(r_mat * np.pi) / (r_mat * np.pi)

    # sinc(0) = 1
    sinc_mat[find_indices_3d(0, 0, 0, original_size, axes, mode)] = 1.0

    if mode == 'half':
        return sinc_mat.reshape((original_size,
                                 original_size,
                                 original_size // 2 + 1))
    elif mode == 'full':
        return sinc_mat.reshape((original_size,
                                 original_size,
                                 original_size))
    else:
        raise ValueError('sinc_3d: mode not defined!')


@jit(nopython=True)
def sinc_3d(original_size, pad):
    padded_size = original_size * pad
    mat = np.empty((original_size, original_size, original_size))
    for i in range(original_size):
        z = i - original_size // 2
        for j in range(original_size):
            y = j - original_size // 2
            for k in range(original_size):
                x = k - original_size // 2

                r = sqrt(x ** 2 + y ** 2 + z ** 2) / padded_size
                if r != 0:
                    mat[i][j][k] = sin(r * np.pi) / (r * np.pi)
                else:
                    mat[i][j][k] = 1.0
    return mat


@jit(nopython=True)
def sinc_correction_3d(vol, pad, trilinear=True):
    size = vol.shape[0]
    for i in range(size):
        l = i - size // 2
        for j in range(size):
            k = j - size // 2
            for t in range(size):
                h = t - size // 2

                r = np.sqrt(h ** 2 + k ** 2 + l ** 2) / (size * pad)
                if r != 0:
                    sinc = np.sin(r * np.pi) / (r * np.pi)
                else:
                    sinc = 1.0
                if trilinear:
                    vol[i][j][t] /= sinc ** 2
                else:
                    vol[i][j][t] /= sinc
