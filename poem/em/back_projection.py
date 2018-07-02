# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from numba import jit
from poem.matrix.rotation import rotation_matrix_euler
from poem.em.ctf import ctf_2d_arr


@jit(nopython=True)
def back_projection(f_2d, f_3d, w_2d, w_3d,
                    phi, theta, psi, x_shift, y_shift,
                    pad=2,
                    ctf=False,
                    defocus_u=-1,
                    defocus_v=-1,
                    defocus_a=-1,
                    cs=-1,
                    apix=-1,
                    kv=-1,
                    ac=-1,
                    b_fac=-1,
                    white=True,
                    trilinear=True):
    # some useful numbers
    size_2d = f_2d.shape[0]
    half_size_2d = size_2d // 2
    size_3d = f_3d.shape[0]
    max_r = size_3d // 2 - 1

    # for each pixel in 2D slice
    for i in range(f_2d.shape[0]):
        # fourier space coordinates
        l = 0
        if i >= half_size_2d:
            k = i - size_2d
        else:
            k = i

        for j in range(f_2d.shape[1]):
            # fourier space coordinates
            h = j

            # rotate the matrix, take padding into account
            rot_mat = np.empty((3, 3))
            rotation_matrix_euler(rot_mat, phi, theta, psi, inv=False)
            x = (rot_mat[0][0] * h +
                 rot_mat[0][1] * k +
                 rot_mat[0][2] * l) * pad
            y = (rot_mat[1][0] * h +
                 rot_mat[1][1] * k +
                 rot_mat[1][2] * l) * pad
            z = (rot_mat[2][0] * h +
                 rot_mat[2][1] * k +
                 rot_mat[2][2] * l) * pad

            # only interpolate within max_r circle
            if x ** 2 + y ** 2 + z ** 2 < max_r ** 2:
                # apply translation
                ph = h * x_shift / size_2d + k * y_shift / size_2d
                f_2d[i][j] *= np.exp(-2 * np.pi * ph * 1.0j)

                # flip phase to center the object in 2D slice
                if (h + k) % 2 == 1:
                    f_2d[i][j] *= -1

                # find friedel's pair if x < 0
                if x < 0:
                    x = -x
                    y = -y
                    z = -z
                    value = np.conj(f_2d[i][j])
                else:
                    value = f_2d[i][j]

                # apply ctf
                if ctf:
                    ctf_value = ctf_2d_arr(h, k, size_2d,
                                           defocus_u, defocus_v, defocus_a,
                                           cs, apix, kv, ac, b_fac)
                    if white:
                        value *= ctf_value
                    else:
                        value *= -ctf_value

                # assign initial weight
                if w_2d[i][j] > 0:
                    weight = w_2d[i][j]
                elif ctf:
                    weight = ctf_value ** 2
                else:
                    weight = 1.0

                # trilinear interpolation
                if trilinear:
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

                # nearest neighbor interpolation
                else:
                    x0 = int(np.round(x))
                    y0 = int(np.round(y))
                    z0 = int(np.round(z))
                    if y0 < 0:
                        y0 += size_3d
                    if z0 < 0:
                        z0 += size_3d

                    f_3d[z0][y0][x0] += value

                    w_3d[z0][y0][x0] += weight


def postprocess_for_reconstruction(f_3d, w_3d, pad=2, trilinear=True):
    # enforce hermitian symmetry, center 3D object, and weighting
    hermitian_center_division(f_3d, w_3d)

    # fourier transform
    vol_pad = np.fft.irfftn(f_3d)

    # initialize output volume
    size = vol_pad.shape[0] // pad
    vol = np.zeros((size, size, size), dtype=np.float32)

    # sinc correction and trimming
    sinc_trim(vol_pad, vol, trilinear)

    return vol


@jit(nopython=True)
def hermitian_center_division(f_3d, w_3d):
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

                # enforce hermitian symmetry on t = 0 plane
                if t == 0 and i <= half_size:

                    # general case
                    if 0 < i < half_size and j > 0:
                        # complex value
                        ave_f = (f_3d[i][j][0] +
                                 np.conj(f_3d[size - i][size - j][0])) / 2
                        f_3d[i][j][0] = ave_f
                        f_3d[size - i][size - j][0] = np.conj(ave_f)

                        # weight
                        ave_w = (w_3d[i][j][0] +
                                 w_3d[size - i][size - j][0]) / 2
                        w_3d[i][j][0] = w_3d[size - i][size - j][0] = ave_w

                    # special point
                    elif (i == 0 or i == half_size) and \
                         (j == 0 or j == half_size):
                        pass

                    # z = 0, special axis, z = size // 2, avoid repetition
                    elif (i == 0 or i == size // 2) and 0 < j < half_size:
                        # complex value
                        ave_f = (f_3d[i][j][0] +
                                 np.conj(f_3d[i][size - j][0])) / 2
                        f_3d[i][j][0] = ave_f
                        f_3d[i][size - j][0] = np.conj(ave_f)

                        # weight
                        ave_w = (w_3d[i][j][0] + w_3d[i][size - j][0]) / 2
                        w_3d[i][j][0] = w_3d[i][size - j][0] = ave_w

                    # y = 0, special axis
                    elif j == 0:
                        # complex value
                        ave_f = (f_3d[i][j][0] +
                                 np.conj(f_3d[size - i][j][0])) / 2
                        f_3d[i][j][0] = ave_f
                        f_3d[size - i][j][0] = np.conj(ave_f)

                        # weight
                        ave_w = (w_3d[i][j][0] + w_3d[size - i][j][0]) / 2
                        w_3d[i][j][0] = w_3d[size - i][j][0] = ave_w

                # center the object in 3D volume
                if (h + k + l) % 2 == 1:
                    f_3d[i][j][t] *= -1

                # weight the pixel (divided by weight)
                if w_3d[i][j][t] > 1e-8:
                    f_3d[i][j][t] /= w_3d[i][j][t]


@jit(nopython=True)
def sinc_trim(vol_pad, vol, trilinear):
    size = vol.shape[0]
    padded_size = vol_pad.shape[0]
    pad_start = (padded_size - size) // 2

    for i in range(size):
        l = i - size // 2
        for j in range(size):
            k = j - size // 2
            for t in range(size):
                h = t - size // 2

                # sinc function
                r = np.sqrt(h ** 2 + k ** 2 + l ** 2) / padded_size
                if r != 0:
                    sinc = np.sin(r * np.pi) / (r * np.pi)
                else:
                    sinc = 1.0

                # trimming and sinc correction
                value = vol_pad[i+pad_start][j+pad_start][t+pad_start]
                if trilinear:
                    vol[i][j][t] = value / sinc ** 2
                else:
                    vol[i][j][t] = value / sinc
