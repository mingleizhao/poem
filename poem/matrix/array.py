# -*- coding: utf-8 -*-

"""
Created on Mon Apr 8 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np


def define_axes(size, axes='center', mode='half'):
    half = size // 2
    if mode == 'half':
        return np.arange(half + 1)
    elif mode == 'full':
        if axes == 'center':
            if size % 2 == 0:
                return np.arange(-half, half)
            else:
                return np.arange(-half, half + 1)
        elif axes == 'fft':
            if size % 2 == 0:
                return np.fft.fftshift(np.arange(-half, half))
            else:
                return np.fft.fftshift(np.arange(-half, half + 1))
        elif axes == 'normal':
            return np.arange(size)
        else:
            raise ValueError('define_axes: axes not defined!')
    else:
        raise ValueError('define_axes: mode not defined!')


def indices_2d(x_size, y_size=None, axes='center', mode='half'):
    if y_size is None:
        y_size = x_size
    x = define_axes(x_size, axes, mode=mode)
    y = define_axes(y_size, axes, mode='full')  # y is always mode full
    # assign coords
    coords = np.zeros(len(x) * len(y), dtype=(np.float, 3))
    coords[:, 0] = np.tile(x, len(y))
    coords[:, 1] = np.repeat(y, len(x))
    return coords


def indices_3d(x_size,
               y_size=None,
               z_size=None,
               axes='center',
               mode='half'):
    if y_size is None and z_size is None:
        z_size = y_size = x_size
    elif y_size is None or z_size is None:
        raise ValueError('indices_3d: provide both y and z size!')
    x = define_axes(x_size, axes, mode=mode)
    y = define_axes(y_size, axes, mode='full')  # y is always mode full
    z = define_axes(z_size, axes, mode='full')  # z is always mode full, too
    # assign coords
    coords = np.zeros(len(x) * len(y) * len(z), dtype=(np.float, 3))
    coords[:, 0] = np.tile(x, len(y) * len(z))
    coords[:, 1] = np.tile(np.repeat(y, len(x)), len(z))
    coords[:, 2] = np.repeat(z, len(y) * len(x))
    return coords


def find_indices_2d(x, y, size, axes='center', mode='half'):
    half_size = size // 2

    # switch axes
    if axes == 'center':
        if mode == 'full':
            x = x + half_size
        y = y + half_size
    elif axes == 'fft':
        if x < 0:
            x = x + size
        if y < 0:
            y = y + size
    elif axes == 'normal':
        pass  # no shift needed
    else:
        raise ValueError('find_indices_2d: axes not defined!')
    # switch mode
    if mode == 'full':
        return int(y * size + x)
    elif mode == 'half':
        return int(y * (half_size + 1) + x)
    else:
        raise ValueError('find_indices_2d: mode not defined!')


def find_indices_3d(x, y, z, size, axes='center', mode='half'):
    half_size = size // 2
    # switch axes
    if axes == 'center':
        if mode == 'full':
            x = x + half_size
        y = y + half_size
        z = z + half_size
    elif axes == 'fft':
        if x < 0:
            x = x + size
        if y < 0:
            y = y + size
        if z < 0:
            z = z + size
    elif axes == 'normal':
        pass  # no shift needed
    else:
        raise ValueError('find_indices_3d: axes not defined!')
    # switch mode
    if mode == 'full':
        return int(z * size * size + y * size + x)
    elif mode == 'half':
        return int(z * size * (half_size + 1) + y * (half_size + 1) + x)
    else:
        raise ValueError('find_indices_3d: mode not defined!')
