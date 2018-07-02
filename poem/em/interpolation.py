# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from numba import jit
from poem.matrix.rotation import rotation_matrix_euler


def interpolate(f_2d, f_3d, phi, theta, psi, pad, interpolator):
    if interpolator == 'trilinear':
        interpolate_tri(f_2d, f_3d, phi, theta, psi, pad)
    elif interpolator == 'nn':
        interpolate_nn(f_2d, f_3d, phi, theta, psi, pad)
    else:
        raise ValueError('interpolate: interpolator not defined!')


@jit(nopython=True)
def interpolate_nn(f_2d, f_3d, phi, theta, psi, pad):
    # sizes
    size_2d = f_2d.shape[0]  # f_3d already padded
    half_size_2d = size_2d // 2
    size_3d = f_3d.shape[0]
    max_r = size_3d // 2 - 1

    for i in range(f_2d.shape[0]):
        if i >= half_size_2d:
            k = i - size_2d
        else:
            k = i
        for j in range(f_2d.shape[1]):
            h = j

            # get rotation matrix
            rot_mat = np.empty((3, 3))
            rotation_matrix_euler(rot_mat, phi, theta, psi, inv=False)

            # rotate h, k, l, taking padding into account
            x = (rot_mat[0][0] * h +
                 rot_mat[0][1] * k) * pad
            y = (rot_mat[1][0] * h +
                 rot_mat[1][1] * k) * pad
            z = (rot_mat[2][0] * h +
                 rot_mat[2][1] * k) * pad

            # only interpolate within max_r circle
            if x ** 2 + y ** 2 + z ** 2 < max_r ** 2:

                # find friedel's pair for x < 0
                if x < 0:
                    neg = True
                    x = -x
                    y = -y
                    z = -z
                else:
                    neg = False

                # nearest neighbor interpolation
                x0 = int(np.round(x))
                y0 = int(np.round(y))
                z0 = int(np.round(z))

                # coordinates shift
                if y0 < 0:
                    y0 += size_3d
                if z0 < 0:
                    z0 += size_3d

                value = f_3d[z0][y0][x0]

                if neg:
                    f_2d[i][j] = np.conj(value)
                else:
                    f_2d[i][j] = value


@jit(nopython=True)
def interpolate_tri(f_2d, f_3d, phi, theta, psi, pad):
    # sizes
    size_2d = f_2d.shape[0]  # f_3d already padded
    half_size_2d = size_2d // 2
    size_3d = f_3d.shape[0]
    max_r = size_3d // 2 - 1

    for i in range(f_2d.shape[0]):
        if i >= half_size_2d:
            k = i - size_2d
        else:
            k = i
        for j in range(f_2d.shape[1]):
            h = j

            # get rotation matrix
            rot_mat = np.empty((3, 3))
            rotation_matrix_euler(rot_mat, phi, theta, psi, inv=False)

            # rotate h, k, l, taking padding into account
            x = (rot_mat[0][0] * h +
                 rot_mat[0][1] * k) * pad
            y = (rot_mat[1][0] * h +
                 rot_mat[1][1] * k) * pad
            z = (rot_mat[2][0] * h +
                 rot_mat[2][1] * k) * pad

            # only interpolate within max_r circle
            if x ** 2 + y ** 2 + z ** 2 < max_r ** 2:

                # find friedel's pair for x < 0
                if x < 0:
                    neg = True
                    x = -x
                    y = -y
                    z = -z
                else:
                    neg = False

                # trilinear interpolation
                x0 = int(np.floor(x))
                x1 = x0 + 1
                xd = x - x0
                y0 = int(np.floor(y))
                y1 = y0 + 1
                yd = y - y0
                z0 = int(np.floor(z))
                z1 = z0 + 1
                zd = z - z0

                # coordinates shift
                if y0 < 0:
                    y0 += size_3d
                if y1 < 0:
                    y1 += size_3d
                if z0 < 0:
                    z0 += size_3d
                if z1 < 0:
                    z1 += size_3d

                d000 = f_3d[z0][y0][x0]
                d100 = f_3d[z0][y0][x1]
                d010 = f_3d[z0][y1][x0]
                d001 = f_3d[z1][y0][x0]
                d110 = f_3d[z0][y1][x1]
                d101 = f_3d[z1][y0][x1]
                d011 = f_3d[z1][y1][x0]
                d111 = f_3d[z1][y1][x1]
                d00 = d000 * (1 - xd) + d100 * xd
                d01 = d001 * (1 - xd) + d101 * xd
                d10 = d010 * (1 - xd) + d110 * xd
                d11 = d011 * (1 - xd) + d111 * xd
                d0 = d00 * (1 - yd) + d10 * yd
                d1 = d01 * (1 - yd) + d11 * yd
                value = d0 * (1 - zd) + d1 * zd

                if neg:
                    f_2d[i][j] = np.conj(value)
                else:
                    f_2d[i][j] = value


def inv_interpolate(f_2d, f_3d, w_2d, w_3d, phi, theta, psi, pad, interpolator):
    if interpolator == 'trilinear':
        inv_interpolate_tri(f_2d, f_3d, w_2d, w_3d, phi, theta, psi, pad)
    elif interpolator == 'nn':
        inv_interpolate_nn(f_2d, f_3d, w_2d, w_3d, phi, theta, psi, pad)
    else:
        raise ValueError('interpolate: interpolator not defined!')


@jit(nopython=True)
def inv_interpolate_nn(f_2d, f_3d, w_2d, w_3d, phi, theta, psi, pad):
    # sizes
    size_2d = f_2d.shape[0]  # f_3d already padded
    half_size_2d = size_2d // 2
    size_3d = f_3d.shape[0]
    max_r = size_3d // 2 - 1

    # for each pixel in 2D slice
    for i in range(f_2d.shape[0]):
        if i >= half_size_2d:
            k = i - size_2d
        else:
            k = i
        for j in range(f_2d.shape[1]):
            h = j

            # rotate the matrix, take padding into account
            rot_mat = np.empty((3, 3))
            rotation_matrix_euler(rot_mat, phi, theta, psi, inv=False)
            x = (rot_mat[0][0] * h +
                 rot_mat[0][1] * k) * pad
            y = (rot_mat[1][0] * h +
                 rot_mat[1][1] * k) * pad
            z = (rot_mat[2][0] * h +
                 rot_mat[2][1] * k) * pad

            # only interpolate within max_r circle
            if x ** 2 + y ** 2 + z ** 2 < max_r ** 2:

                # find friedel's pair if x < 0
                if x < 0:
                    x = -x
                    y = -y
                    z = -z
                    value = np.conj(f_2d[i][j])
                else:
                    value = f_2d[i][j]

                # assign initial weight
                if w_2d[i][j] > 0:
                    weight = w_2d[i][j]
                else:
                    weight = 1.0

                # inverse nearest neighbor interpolation
                x0 = int(np.round(x))
                y0 = int(np.round(y))
                z0 = int(np.round(z))
                if y0 < 0:
                    y0 += size_3d
                if z0 < 0:
                    z0 += size_3d

                f_3d[z0][y0][x0] += value

                w_3d[z0][y0][x0] += weight


@jit(nopython=True)
def inv_interpolate_tri(f_2d, f_3d, w_2d, w_3d, phi, theta, psi, pad):
    # sizes
    size_2d = f_2d.shape[0]  # f_3d already padded
    half_size_2d = size_2d // 2
    size_3d = f_3d.shape[0]
    max_r = size_3d // 2 - 1

    # for each pixel in 2D slice
    for i in range(f_2d.shape[0]):
        if i >= half_size_2d:
            k = i - size_2d
        else:
            k = i
        for j in range(f_2d.shape[1]):
            h = j

            # rotate the matrix, take padding into account
            rot_mat = np.empty((3, 3))
            rotation_matrix_euler(rot_mat, phi, theta, psi, inv=False)
            x = (rot_mat[0][0] * h +
                 rot_mat[0][1] * k) * pad
            y = (rot_mat[1][0] * h +
                 rot_mat[1][1] * k) * pad
            z = (rot_mat[2][0] * h +
                 rot_mat[2][1] * k) * pad

            # only interpolate within max_r circle
            if x ** 2 + y ** 2 + z ** 2 < max_r ** 2:

                # find friedel's pair if x < 0
                if x < 0:
                    x = -x
                    y = -y
                    z = -z
                    value = np.conj(f_2d[i][j])
                else:
                    value = f_2d[i][j]

                # assign initial weight
                if w_2d[i][j] > 0:
                    weight = w_2d[i][j]
                else:
                    weight = 1.0

                # inverse trilinear interpolation
                x0 = int(np.floor(x))
                x1 = x0 + 1
                xd = x - x0
                y0 = int(np.floor(y))
                y1 = y0 + 1
                yd = y - y0
                z0 = int(np.floor(z))
                z1 = z0 + 1
                zd = z - z0

                # calculate distance weights
                d0 = 1 - zd
                d1 = zd
                d00 = d0 * (1 - yd)
                d01 = d1 * (1 - yd)
                d10 = d0 * yd
                d11 = d1 * yd
                d000 = d00 * (1 - xd)
                d100 = d00 * xd
                d010 = d10 * (1 - xd)
                d001 = d01 * (1 - xd)
                d110 = d10 * xd
                d101 = d01 * xd
                d011 = d11 * (1 - xd)
                d111 = d11 * xd

                # coordinates shift
                if y0 < 0:
                    y0 += size_3d
                if y1 < 0:
                    y1 += size_3d
                if z0 < 0:
                    z0 += size_3d
                if z1 < 0:
                    z1 += size_3d

                f_3d[z0][y0][x0] += d000 * value
                f_3d[z0][y0][x1] += d001 * value
                f_3d[z0][y1][x0] += d010 * value
                f_3d[z1][y0][x0] += d100 * value
                f_3d[z0][y1][x1] += d011 * value
                f_3d[z1][y0][x1] += d101 * value
                f_3d[z1][y1][x0] += d110 * value
                f_3d[z1][y1][x1] += d111 * value

                w_3d[z0][y0][x0] += d000 * weight
                w_3d[z0][y0][x1] += d001 * weight
                w_3d[z0][y1][x0] += d010 * weight
                w_3d[z1][y0][x0] += d100 * weight
                w_3d[z0][y1][x1] += d011 * weight
                w_3d[z1][y0][x1] += d101 * weight
                w_3d[z1][y1][x0] += d110 * weight
                w_3d[z1][y1][x1] += d111 * weight
