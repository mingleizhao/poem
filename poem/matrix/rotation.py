# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


from math import radians
from math import cos
from math import sin
from math import sqrt
from numba import jit


@jit(nopython=True)
def rotation_matrix_euler(mat, phi, theta, psi, inv=False):
    # convert degrees to radians
    alpha = radians(phi)
    beta = radians(theta)
    gamma = radians(psi)

    # calculate sin and cos function
    c1 = cos(alpha)
    c2 = cos(beta)
    c3 = cos(gamma)
    s1 = sin(alpha)
    s2 = sin(beta)
    s3 = sin(gamma)

    # define rotation matrix (zyz convention)
    if inv:
        mat[0][0] = c1 * c2 * c3 - s1 * s3
        mat[0][1] = c1 * s3 + c2 * c3 * s1
        mat[0][2] = -c3 * s2
        mat[1][0] = -c3 * s1 - c1 * c2 * s3
        mat[1][1] = c1 * c3 - c2 * s1 * s3
        mat[1][2] = s2 * s3
        mat[2][0] = c1 * s2
        mat[2][1] = s1 * s2
        mat[2][2] = c2
    else:
        mat[0][0] = c1 * c2 * c3 - s1 * s3
        mat[0][1] = -c3 * s1 - c1 * c2 * s3
        mat[0][2] = c1 * s2
        mat[1][0] = c1 * s3 + c2 * c3 * s1
        mat[1][1] = c1 * c3 - c2 * s1 * s3
        mat[1][2] = s1 * s2
        mat[2][0] = -c3 * s2
        mat[2][1] = s2 * s3
        mat[2][2] = c2

    # original matrix
    # mat = [[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
    #        [c1*s3+c2*c3*s1,  c1*c3-c2*s1*s3, s1*s2],
    #        [        -c3*s2,           s2*s3,    c2]])


@jit(nopython=True)
def rotation_matrix_2d(mat, theta, inv=False):
    # convert degrees to radians
    degree = radians(theta)

    # calculate sin and cos function
    c = cos(degree)
    s = sin(degree)

    # define rotation matrix
    if inv:
        mat[0][0] = c
        mat[0][1] = s
        mat[1][0] = -s
        mat[1][1] = c
    else:
        mat[0][0] = c
        mat[0][1] = -s
        mat[1][0] = s
        mat[1][1] = c

    # original matrix
    # mat = [[c, -s],
    #        [s,  c]])


def rotation_matrix_axis(mat, theta, inv=False, axis='z'):
    # convert degrees to radians
    degree = radians(theta)
    if inv:
        degree *= -1

    # calculate sin and cos function
    c = cos(degree)
    s = sin(degree)

    # define rotation matrix
    if axis == 'z':
        mat[0][0] = c
        mat[0][1] = -s
        mat[0][2] = 0
        mat[1][0] = s
        mat[1][1] = c
        mat[1][2] = 0
        mat[2][0] = 0
        mat[2][1] = 0
        mat[2][2] = 1
    elif axis == 'y':
        mat[0][0] = c
        mat[0][1] = 0
        mat[0][2] = s
        mat[1][0] = 0
        mat[1][1] = 1
        mat[1][2] = 0
        mat[2][0] = -s
        mat[2][1] = 0
        mat[2][2] = c
    elif axis == 'x':
        mat[0][0] = 1
        mat[0][1] = 0
        mat[0][2] = 0
        mat[1][0] = 0
        mat[1][1] = c
        mat[1][2] = -s
        mat[2][0] = 0
        mat[2][1] = s
        mat[2][2] = c
    else:
        raise ValueError('rotation_matrix_axis: unknown axis!')


def rotation_matrix_vector(mat, theta, inv=False, vec=(0, 0, 1)):
    # convert degrees to radians
    degree = radians(theta)
    if inv:
        degree *= -1

    # calculate sin and cos function
    c = cos(degree)
    s = sin(degree)

    # calculate unit vector
    norm = sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    ux = vec[0] / norm
    uy = vec[1] / norm
    uz = vec[2] / norm

    # define rotation matrix
    mat[0][0] = c + ux ** 2 * (1 - c)
    mat[0][1] = ux * uy * (1 - c) - uz * s
    mat[0][2] = ux * uz * (1 - c) + uy * s
    mat[1][0] = uy * ux * (1 - c) + uz * s
    mat[1][1] = c + uy ** 2 * (1 - c)
    mat[1][2] = uy * uz * (1 - c) - ux * s
    mat[2][0] = uz * ux * (1 - c) - uy * s
    mat[2][1] = uz * uy * (1 - c) + ux * s
    mat[2][2] = c + uz ** 2 * (1 - c)


def rotation_matrix_align_vector_to_axis(mat, vec, axis='z'):
    # calculate unit vector
    norm = sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    ux = vec[0] / norm
    uy = vec[1] / norm
    uz = vec[2] / norm

    # define rotation matrix
    if axis == 'z':
        proj_mod = sqrt(uy ** 2 + uz ** 2)
        mat[0][0] = proj_mod
        mat[0][1] = -ux * uy / proj_mod
        mat[0][2] = -ux * uz / proj_mod
        mat[1][0] = 0
        mat[1][1] = uz / proj_mod
        mat[1][2] = -uy / proj_mod
        mat[2][0] = ux
        mat[2][1] = uy
        mat[2][2] = uz
    elif axis == 'y':
        proj_mod = sqrt(uy ** 2 + uz ** 2)
        mat[0][0] = proj_mod
        mat[0][1] = -ux * uy / proj_mod
        mat[0][2] = -ux * uz / proj_mod
        mat[1][0] = 0
        mat[1][1] = uz / proj_mod
        mat[1][2] = -uy / proj_mod
        mat[2][0] = ux
        mat[2][1] = uy
        mat[2][2] = uz
    elif axis == 'x':
        proj_mod = sqrt(uy ** 2 + uz ** 2)
        mat[0][0] = proj_mod
        mat[0][1] = -ux * uy / proj_mod
        mat[0][2] = -ux * uz / proj_mod
        mat[1][0] = 0
        mat[1][1] = uz / proj_mod
        mat[1][2] = -uy / proj_mod
        mat[2][0] = ux
        mat[2][1] = uy
        mat[2][2] = uz
    else:
        raise ValueError('rotation_matrix_align_vector_to_axis: unknown axis!')
