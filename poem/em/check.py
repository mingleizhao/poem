# -*- coding: utf-8 -*-
"""
Created on June 18th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


def sanity_check_volume(vol):
    if vol.ndim != 3:
        raise ValueError('sanity_check_volume: volume dimension not right!')
    if vol.shape[0] != vol.shape[1] or vol.shape[0] != vol.shape[2]:
        raise ValueError('sanity_check_volume: volume has to be cubic!')
    if vol.shape[0] % 2 != 0:
        raise ValueError('sanity_check_volume: volume edges have to be even!')


def sanity_check_stack(stack):
    if stack.ndim != 3:
        raise ValueError('sanity_check_stack: input data not a stack!')
    if stack.shape[1] != stack.shape[2]:
        raise ValueError('sanity_check_stack: images must be square!')
    if stack.shape[1] % 2 != 0:
        raise ValueError('sanity_check_stack: image edges have to be even!')


def sanity_check_image(img):
    if img.ndim != 2:
        raise ValueError('sanity_check_image: image dimension not right!')
    if img.shape[0] != img.shape[1]:
        raise ValueError('sanity_check_image: image has to be square!')
    if img.shape[0] % 2 != 0:
        raise ValueError('sanity_check_image: image edges have to be even!')


def sanity_check_f_2d(f_2d):
    if f_2d.ndim != 2:
        raise ValueError('sanity_check_f_2d: fft dimension not right!')
    if (f_2d.shape[0] // 2 + 1) != f_2d.shape[1]:
        raise ValueError('sanity_check_f_2d: fft sizes not right!')
    if f_2d.shape[0] % 2 != 0:
        raise ValueError('sanity_check_f_2d: fft longer edge has to be even!')


def sanity_check_f_3d(f_3d):
    if f_3d.ndim != 3:
        raise ValueError('sanity_check_f_3d: fft dimension not right!')
    if f_3d.shape[0] != f_3d.shape[1] or \
       (f_3d.shape[0] // 2 + 1) != f_3d.shape[2]:
        raise ValueError('sanity_check_f_3d: fft sizes not right!')
    if f_3d.shape[0] % 2 != 0:
        raise ValueError('sanity_check_f_3d: fft longer edges have to be even!')
