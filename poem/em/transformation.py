# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from numba import jit
from poem.matrix.rotation import rotation_matrix_2d
from poem.matrix.rotation import rotation_matrix_euler


def rotate_2d(x, y, theta, pad):
    rot_mat = rotation_matrix_2d(theta, inv=False)
    x_ = (rot_mat[0][0] * x + rot_mat[0][1] * y) * pad
    y_ = (rot_mat[1][0] * x + rot_mat[1][1] * y) * pad
    return x_, y_


def rotate_3d(x, y, z, phi, theta, psi, pad):
    rot_mat = rotation_matrix_euler(phi, theta, psi, inv=False)
    x_ = (rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2] * z) * pad
    y_ = (rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2] * z) * pad
    z_ = (rot_mat[2][0] * x + rot_mat[2][1] * y + rot_mat[2][2] * z) * pad
    return x_, y_, z_


@jit(nopython=True)
def translate_2d(f_2d, x, y):
    size = f_2d.shape[0]
    half_size = size // 2
    if x != 0 or y != 0:
        for i in range(size):
            if i >= half_size:
                k = i - size
            else:
                k = i
            for j in range(half_size + 1):
                h = j

                ph = h * x / size + k * y / size
                f_2d[i][j] *= np.exp(-2 * np.pi * ph * 1.0j)


@jit(nopython=True)
def translate_3d(f_3d, x, y, z):
    size = f_3d.shape[0]
    half_size = size // 2
    if x != 0 or y != 0 or z != 0:
        for i in range(size):
            if i >= half_size:
                l = i - size
            else:
                l = i
            for j in range(size):
                if j >= half_size:
                    k = j - size
                else:
                    k = j
                for t in range(half_size + 1):
                    h = t

                    ph = h * x / size + k * y / size + l * z / size
                    f_3d[i][j][t] *= np.exp(-2 * np.pi * ph * 1.0j)


@jit(nopython=True)
def center_3d(f_3d):
    size = f_3d.shape[0]
    half_size = size // 2
    for i in range(size):
        if i >= half_size:
            l = i - size
        else:
            l = i
        for j in range(size):
            if j >= half_size:
                k = j - size
            else:
                k = j
            for t in range(half_size + 1):
                h = t

                if (h + k + l) % 2 == 1:
                    f_3d[i][j][t] *= -1


@jit(nopython=True)
def center_2d(f_2d):
    size = f_2d.shape[0]
    half_size = size // 2
    for i in range(size):
        if i >= half_size:
            k = i - size
        else:
            k = i
        for j in range(half_size + 1):
                h = j

                if (h + k) % 2 == 1:
                    f_2d[i][j] *= -1
