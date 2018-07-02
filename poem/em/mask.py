# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


from numba import jit


@jit(nopython=True)
def apply_mask_2d(img, mask):
    if img.ndim != 2:
        raise ValueError('apply_mask_2d: not a valid image!')
    if img.shape != mask.shape:
        raise ValueError('apply_mask_2d: image and mask must be in the same '
                         'dimension!')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] *= mask[i][j]


@jit(nopython=True)
def apply_mask_3d(vol, mask):
    if vol.ndim != 3:
        raise ValueError('apply_mask_3d: not a valid volume!')
    if vol.shape != mask.shape:
        raise ValueError('apply_mask_3d: volume and mask must be in the same '
                         'dimension!')
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                vol[i][j][k] *= mask[i][j][k]
