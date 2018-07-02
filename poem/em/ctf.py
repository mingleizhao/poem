# -*- coding: utf-8 -*-

"""
Created on Mon Mar 30 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com


"""

from math import radians
import numpy as np
from numba import vectorize
from poem.matrix.array import indices_2d


def ctf_1d(x,  # unit: pixel
           defocus,  # unit: Angstrom
           cs,  # unit: mm
           apix=1.0,  # unit: Angstrom/pixel
           kv=200,  # unit: kilo volt
           ac=0.1,
           b_fac=0.0):
    local_v = kv * 1e3  # unit: volt
    local_cs = cs * 1e7  # unit: A
    wavelength = 12.2643247 / np.sqrt(local_v * (1.0 + local_v * 0.978466e-6))
    # print('wavelength = ', wavelength)
    wavelength_3 = wavelength ** 3
    # construct vector
    if x % 2 == 0:
        v = np.arange(x // 2 + 1)
    else:
        v = np.arange((x + 1) // 2)
    x_len = x * apix
    k = v / x_len  # reciprocal vector k
    k_2 = k * k
    k_4 = k_2 * k_2
    # phase shift
    ph = np.pi / 2 * local_cs * wavelength_3 * k_4 - \
         np.pi * wavelength * defocus * k_2
    # print('phase_shift = ', ph)
    # ctf
    ctf = -np.sqrt(1.0 - ac * ac) * np.sin(ph) + ac * np.cos(ph)
    # print('ctf = ', ctf)
    # damp ctf based on b_fac
    if b_fac > 0:
        ctf *= np.exp(-b_fac / 4.0 * k_2)
    return ctf


def ctf_2d(x,  # unit: pixel
           y,  # unit: pixel
           defocus_u,  # unit: Angstrom
           defocus_v,  # unit: Angstrom
           defocus_a,  # in degree
           cs,  # unit: mm
           apix_x=1.0,  # unit: Angstrom/pixel
           apix_y=None,
           kv=200,  # unit: kilo voltage
           ac=0.1,  # arbitrary unit
           b_fac=0.0,
           axes='center',
           mode='half',
           return_matrix=True):

    # convert to correct unit
    azimuthal_a = radians(defocus_a)  # convert to rad
    local_v = kv * 1e3  # unit: volt
    local_cs = cs * 1e7  # unit: A

    # wavelength
    wavelength = 12.2643247 / np.sqrt(local_v * (1.0 + local_v * 0.978466e-6))
    wavelength_3 = wavelength ** 3

    # if apix_y is not provided assume sampling rate is the same for x and y
    if apix_y is None:
        apix_y = apix_x
    x_len = x * apix_x
    y_len = y * apix_y

    # construct the coordinate array
    arr = indices_2d(x, y, axes=axes, mode=mode)

    # reciprocal k vector
    k_angle = np.arctan2(arr[:, 1], arr[:, 0])
    k = np.sqrt((arr[:, 0] / x_len) ** 2 + (arr[:, 1] / y_len) ** 2)
    k_2 = k * k
    k_4 = k_2 * k_2
    a_shift = k_angle - azimuthal_a

    # defocus
    defocus = defocus_u * np.cos(a_shift) * np.cos(a_shift) + \
              defocus_v * np.sin(a_shift) * np.sin(a_shift)

    # phase shift
    ph = np.pi / 2 * local_cs * wavelength_3 * k_4 - \
         np.pi * wavelength * defocus * k_2

    # ctf
    ctf = -np.sqrt(1.0 - ac * ac) * np.sin(ph) + ac * np.cos(ph)

    # damp ctf based on b_fac
    if b_fac > 0:
        ctf *= np.exp(-b_fac / 4.0 * k_2)

    if return_matrix:
        if mode == 'half':
            shape = (y, x // 2 + 1)
        elif mode == 'full':
            shape = (y, x)
        else:
            raise ValueError('ctf_2d: mode not defined!')
        return ctf.reshape(shape)
    else:
        return ctf


@vectorize
def ctf_2d_arr(x,  # x coordinates
               y,  # y coordinates
               size,  # unit: pixel
               defocus_u,  # unit: Angstrom
               defocus_v,  # unit: Angstrom
               defocus_a,  # in degree
               cs,  # unit: mm
               apix=1.0,  # unit: Angstrom/pixel
               kv=200,  # unit: kilo voltage
               ac=0.1,  # arbitrary unit
               b_fac=0.0):
    # physical length
    length = size * apix

    # convert to correct unit
    azimuthal_a = radians(defocus_a)  # convert to rad
    local_v = kv * 1e3  # unit: volt
    local_cs = cs * 1e7  # unit: A

    # wavelength
    wavelength = 12.2643247 / np.sqrt(local_v * (1.0 + local_v * 0.978466e-6))
    wavelength_3 = wavelength ** 3

    # reciprocal k vector
    k_angle = np.arctan2(y, x)
    k = np.sqrt((x / length) ** 2 + (y / length) ** 2)
    k_2 = k * k
    k_4 = k_2 * k_2
    a_shift = k_angle - azimuthal_a

    # defocus
    defocus = defocus_u * np.cos(a_shift) * np.cos(a_shift) + \
              defocus_v * np.sin(a_shift) * np.sin(a_shift)

    # phase shift
    ph = np.pi / 2 * local_cs * wavelength_3 * k_4 - \
         np.pi * wavelength * defocus * k_2

    # ctf
    ctf = -np.sqrt(1.0 - ac * ac) * np.sin(ph) + ac * np.cos(ph)

    # damp ctf based on b_fac
    if b_fac > 0:
        ctf *= np.exp(-b_fac / 4.0 * k_2)

    # print(x, y, size, ctf)
    return ctf
