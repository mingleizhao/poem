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
from poem.em.check import sanity_check_volume


@jit(nopython=True)
def sinc_pad(vol_pad, vol, trilinear):
    size = vol.shape[0]
    padded_size = vol_pad.shape[0]
    pad_start = (padded_size - size) // 2

    for i in range(size):
        l = i - size // 2
        for j in range(size):
            k = j - size // 2
            for t in range(size):
                h = t - size // 2

                # sinc correct
                r = np.sqrt(h ** 2 + k ** 2 + l ** 2) / padded_size
                if r != 0:
                    sinc = np.sin(r * np.pi) / (r * np.pi)
                else:
                    sinc = 1.0
                if trilinear:
                    value = vol[i][j][t] / sinc ** 2
                else:
                    value = vol[i][j][t] / sinc

                # padding
                vol_pad[i+pad_start][j+pad_start][t+pad_start] = value


def preprocess_for_projection(vol, pad=2, trilinear=True):
    # sanity check
    sanity_check_volume(vol)

    # initialize padded volume
    padded_size = vol.shape[0] * pad
    vol_pad = np.zeros((padded_size, padded_size, padded_size))

    # sinc correction and padding
    sinc_pad(vol_pad, vol, trilinear)

    # fourier transform
    f_3d = np.fft.rfftn(vol_pad)

    return f_3d


@jit(nopython=True)
def projection(f_2d, f_3d,
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
                    neg = True
                    x = -x
                    y = -y
                    z = -z
                else:
                    neg = False

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
                    if y0 < 0:
                        y0 += size_3d
                    if y1 < 0:
                        y1 += size_3d
                    if z0 < 0:
                        z0 += size_3d
                    if z1 < 0:
                        z1 += size_3d

                    # flip phase to center the object in 3D volume
                    if (z0 + y0 + x0) % 2 == 1:
                        flip = -1
                    else:
                        flip = 1

                    d000 = f_3d[z0][y0][x0] * flip
                    d100 = f_3d[z0][y0][x1] * (-flip)
                    d010 = f_3d[z0][y1][x0] * (-flip)
                    d001 = f_3d[z1][y0][x0] * (-flip)
                    d110 = f_3d[z0][y1][x1] * flip
                    d101 = f_3d[z1][y0][x1] * flip
                    d011 = f_3d[z1][y1][x0] * flip
                    d111 = f_3d[z1][y1][x1] * (-flip)
                    d00 = d000 * (1 - xd) + d100 * xd
                    d01 = d001 * (1 - xd) + d101 * xd
                    d10 = d010 * (1 - xd) + d110 * xd
                    d11 = d011 * (1 - xd) + d111 * xd
                    d0 = d00 * (1 - yd) + d10 * yd
                    d1 = d01 * (1 - yd) + d11 * yd
                    value = d0 * (1 - zd) + d1 * zd

                # nearest neighbor interpolation
                else:
                    x0 = int(np.round(x))
                    y0 = int(np.round(y))
                    z0 = int(np.round(z))
                    if y0 < 0:
                        y0 += size_3d
                    if z0 < 0:
                        z0 += size_3d

                    # flip phase to center the object in 3D volume
                    if (z0 + y0 + x0) % 2 == 1:
                        flip = -1
                    else:
                        flip = 1

                    value = f_3d[z0][y0][x0] * flip

                # conjugate if the pixel is friedel's pair
                if neg:
                    f_2d[i][j] = np.conj(value)
                else:
                    f_2d[i][j] = value

                # apply translation
                ph = h * x_shift / size_2d + k * y_shift / size_2d
                f_2d[i][j] *= np.exp(-2 * np.pi * ph * 1.0j)

                # apply ctf
                if ctf:
                    ctf_value = ctf_2d_arr(h, k, size_2d,
                                           defocus_u, defocus_v, defocus_a,
                                           cs, apix, kv, ac, b_fac)
                    if white:
                        f_2d[i][j] *= ctf_value
                    else:
                        f_2d[i][j] *= -ctf_value

                # flip phase to center the object in 2D slice
                if (h + k) % 2 == 1:
                    f_2d[i][j] *= -1
