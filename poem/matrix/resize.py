# -*- coding: utf-8 -*-

"""
Created on Mon Apr 8 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from numba import jit


def pad_2d_general(img, pad=1):
    # sanity check
    if len(img.shape) != 2:
        raise ValueError('pad_2d: image dimension not right!')
    if pad > 1:
        size = max(img.shape) * pad
        diff_0 = size - img.shape[0]
        diff_1 = size - img.shape[1]
        before_0 = int(diff_0 / 2)
        after_0 = int(diff_0 - before_0)
        before_1 = int(diff_1 / 2)
        after_1 = int(diff_1 - before_1)
        img_pad = np.pad(img,
                         [(before_0, after_0), (before_1, after_1)],
                         mode='constant')
    else:
        img_pad = img

    return img_pad


def trim_2d_general(img, trim):
    # sanity check
    if len(img.shape) != 2:
        raise ValueError('trim_2d: image dimension not right!')
    if img.shape[0] != img.shape[1]:
        raise ValueError('trim_2d: image not square!')
    if trim > 1:
        size = img.shape[0] // trim
        start = (img.shape[0] - size) // 2
        end = start + size
        return img[start:end, start:end]
    else:
        return img


def pad_3d_general(v, pad=1):
    # sanity check
    if len(v.shape) != 3:
        raise ValueError('pad_3d: volume dimension not right!')
    if pad > 1:
        size = max(v.shape) * pad
        diff_0 = size - v.shape[0]
        diff_1 = size - v.shape[1]
        diff_2 = size - v.shape[2]
        before_0 = int(diff_0 / 2)
        after_0 = int(diff_0 - before_0)
        before_1 = int(diff_1 / 2)
        after_1 = int(diff_1 - before_1)
        before_2 = int(diff_2 / 2)
        after_2 = int(diff_2 - before_2)
        v_pad = np.pad(v, [(before_0, after_0),
                           (before_1, after_1),
                           (before_2, after_2)], mode='constant')
    else:
        v_pad = v
    return v_pad


def trim_3d_general(v, trim):
    # sanity check
    if len(v.shape) != 3:
        raise ValueError('trim_3d: volume dimension not right!')
    if v.shape[0] != v.shape[1] != v.shape[2]:
        raise ValueError('trim_3d: volume not cubic!')
    if trim > 1:
        size = v.shape[0] // trim
        start = (v.shape[0] - size) // 2
        end = start + size
        return v[start:end, start:end, start:end]
    else:
        return v


@jit(nopython=True)
def pad_2d(img_pad, img):
    y_pad = img_pad.shape[0]
    y = img.shape[0]
    x_pad = img_pad.shape[1]
    x = img.shape[1]
    start_y = (y_pad - y) // 2
    start_x = (x_pad - x) // 2

    for i in range(y):
        for j in range(x):
                img_pad[i+start_y][j+start_x] = img[i][j]


@jit(nopython=True)
def pad_3d(v_pad, v):
    z_pad = v_pad.shape[0]
    z = v.shape[0]
    y_pad = v_pad.shape[1]
    y = v.shape[1]
    x_pad = v_pad.shape[2]
    x = v.shape[2]
    start_z = (z_pad - z) // 2
    start_y = (y_pad - y) // 2
    start_x = (x_pad - x) // 2

    for i in range(z):
        for j in range(y):
            for k in range(x):
                v_pad[i+start_z][j+start_y][k+start_x] = v[i][j][k]


@jit(nopython=True)
def trim_2d(img_pad, img):
    y_pad = img_pad.shape[0]
    y = img.shape[0]
    x_pad = img_pad.shape[1]
    x = img.shape[1]
    start_y = (y_pad - y) // 2
    start_x = (x_pad - x) // 2

    for i in range(y):
        for j in range(x):
            img[i][j] = img_pad[i+start_y][j+start_x]


@jit(nopython=True)
def trim_3d(v_pad, v):
    z_pad = v_pad.shape[0]
    z = v.shape[0]
    y_pad = v_pad.shape[1]
    y = v.shape[1]
    x_pad = v_pad.shape[2]
    x = v.shape[2]
    start_z = (z_pad - z) // 2
    start_y = (y_pad - y) // 2
    start_x = (x_pad - x) // 2

    for i in range(z):
        for j in range(y):
            for k in range(x):
                v[i][j][k] = v_pad[i+start_z][j+start_y][k+start_x]
