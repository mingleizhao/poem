# -*- coding: utf-8 -*-

"""
Created on Mon June 20 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from numba import jit


@jit(nopython=True)
def enforce_hermitian_symmetry_2d(f_2d):
    size = f_2d.shape[0]
    for i in range(1, size // 2):
            average = (f_2d[i][0] + np.conj(f_2d[size - i][0])) / 2
            f_2d[i][0] = average
            f_2d[size - i][0] = np.conj(average)


@jit(nopython=True)
def enforce_hermitian_symmetry_3d(f_3d):
    size = f_3d.shape[0]
    for i in range(1, size):
        for j in range(1, size // 2):
            average = (f_3d[i][j][0] + np.conj(f_3d[size - i][size - j][0])) / 2
            f_3d[i][j][0] = average
            f_3d[size - i][size - j][0] = np.conj(average)

    # special positions
    for i in range(1, size // 2):
        # z = 0
        average = ((f_3d[0][i][0]) + np.conj(f_3d[0][size - i][0])) / 2
        f_3d[0][i][0] = average
        f_3d[0][size - i][0] = np.conj(average)
        # y = 0
        average = ((f_3d[i][0][0]) + np.conj(f_3d[size - i][0][0])) / 2
        f_3d[i][0][0] = average
        f_3d[size - i][0][0] = np.conj(average)
        # y = size // 2
        average = ((f_3d[i][size // 2][0]) +
                   np.conj(f_3d[size - i][size // 2][0])) / 2
        f_3d[i][size // 2][0] = average
        f_3d[size - i][size // 2][0] = np.conj(average)


class PointGroup:
    def __init__(self, pg_name):
        self.pg_category, self.pg_order = self.parse_pg_symmetry(pg_name)

    def parse_pg_symmetry(self, pg_name):
        pg_name = pg_name.upper()
        pg_length = len(pg_name)

        if pg_length > 4 or pg_length == 0:
            raise ValueError('PointGroup: invalid point group symbol!')
        elif pg_length == 1:
            # T, O, or I
            if pg_name == 'T':
                return 'PG_T', None
            elif pg_name == 'O':
                return 'PG_O', None
            elif pg_name == 'I':
                return 'PG_I', None
            else:
                raise ValueError('PointGroup: invalid point group symbol!')
        elif pg_length == 2:
            if pg_name[1].isalpha():
                # Cs, Ci, Th, Td, Oh, Ih,
                if pg_name == 'CS':
                    return 'PG_CS', None
                elif pg_name == 'CI':
                    return 'PG_CI', None
                elif pg_name == 'TH':
                    return 'PG_TH', None
                elif pg_name == 'TD':
                    return 'PG_TD', None
                elif pg_name == 'OH':
                    return 'PG_OH', None
                elif pg_name == 'IH':
                    return 'PG_IH', None
                else:
                    raise ValueError('PointGroup: invalid point group symbol!')
            elif pg_name[1].isdigit():
                # Cn, Dn, Sn
                pg_order = int(pg_name[1])
                if pg_order == 0:
                    raise ValueError('PointGroup: order cannot be zero!')
                if pg_name[0] == 'C':
                    return 'PG_CN', pg_order
                elif pg_name[0] == 'D' and pg_order != 1:
                    return 'PG_DN', pg_order
                elif pg_name[0] == 'S':
                    if pg_order % 2 == 1:
                        raise ValueError(
                            'PointGroup: order must be even!')
                    else:
                        return 'PG_SN', pg_order
                else:
                    raise ValueError('PointGroup: invalid point group '
                                     'category!')
            else:
                raise ValueError('PointGroup: invalid point group symbol!')
        elif pg_length == 3:
            if pg_name[0] == 'C':
                if pg_name[2].isalpha() and pg_name[1].isdigit():
                    pg_order = int(pg_name[1])
                    if pg_order == 0 or pg_order == 1:
                        raise ValueError('PointGroup: order cannot be zero '
                                         'or one!')
                    if pg_name[2] == 'V':
                        return 'PG_CNV', pg_order
                    elif pg_name[2] == 'H':
                        return 'PG_CNH', pg_order
                    else:
                        raise ValueError('PointGroup: invalid point group '
                                         'symbol!')
                elif pg_name[1].isdigit() and pg_name[2].isdigit():
                    pg_order = int(pg_name[1:])
                    if pg_order // 10 == 0:
                        raise ValueError('PointGroup: wrong order!')
                    return 'PG_CN', pg_order
                else:
                    raise ValueError('PointGroup: invalid point group symbol!')
            elif pg_name[0] == 'D':
                if pg_name[2].isalpha() and pg_name[1].isdigit():
                    pg_order = int(pg_name[1])
                    if pg_order == 0 or pg_order == 1:
                        raise ValueError('PointGroup: order cannot be zero or '
                                         'one!')
                    if pg_name[2] == 'D':
                        return 'PG_DND', pg_order
                    elif pg_name[2] == 'H':
                        return 'PG_DNH', pg_order
                    else:
                        raise ValueError('PointGroup: invalid point group '
                                         'symbol!')
                elif pg_name[1].isdigit() and pg_name[2].isdigit():
                    pg_order = int(pg_name[1:])
                    if pg_order // 10 == 0:
                        raise ValueError('PointGroup: wrong order!')
                    return 'PG_DN', pg_order
                else:
                    raise ValueError('PointGroup: invalid point group symbol!')
            elif pg_name[0] == 'S':
                if pg_name[1].isdigit() and pg_name[2].isdigit():
                    pg_order = int(pg_name[1:])
                    if pg_order // 10 == 0:
                        raise ValueError('PointGroup: wrong order!')
                    elif pg_order % 2 == 1:
                        raise ValueError('PointGroup: order cannot be even!')
                    else:
                        return 'PG_SN', pg_order
                else:
                    raise ValueError('PointGroup: invalid point group symbol!')
            else:
                raise ValueError('PointGroup: invalid point group '
                                 'category!')
        else:  # pg_length == 4
            if pg_name[1].isdigit() and \
               pg_name[2].isdigit() and \
               pg_name[3].isalpha():
                pg_order = int(pg_name[1:3])
                if pg_order // 10 == 0:
                    raise ValueError('PointGroup: wrong order!')
                if pg_name[0] == 'C':
                    if pg_name[3] == 'V':
                        return 'PG_CNV', pg_order
                    elif pg_name[3] == 'H':
                        return 'PG_CNH', pg_order
                    else:
                        raise ValueError(
                            'PointGroup: invalid point group symbol!')
                elif pg_name[0] == 'D':
                    if pg_name[3] == 'D':
                        return 'PG_DND', pg_order
                    elif pg_name[3] == 'H':
                        return 'PG_DNH', pg_order
                    else:
                        raise ValueError(
                            'PointGroup: invalid point group symbol!')
                else:
                    raise ValueError('PointGroup: invalid point group '
                                     'category!')
            else:
                raise ValueError('PointGroup: invalid point group symbol!')
