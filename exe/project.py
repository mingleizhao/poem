# -*- coding: utf-8 -*-
"""
Created on June 7th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from poem.data.mrc import Mrc
from poem.data.star import RelionStar
from poem.data.star import read_relion_transformation
from poem.data.star import read_relion_ctf
from poem.em.projection import preprocess_for_projection
from poem.em.projection import projection


star_file = '/Users/minglei/PROJECTS/poem/test/70S.star'
map_file = '/Users/minglei/PROJECTS/poem/test/70S_1_r1.mrc'
stack_file = '/Users/minglei/PROJECTS/poem/test/70S_projected.mrc'

pad = 2
res = 1.0
apix = 2.52
trilinear = True
white = True

map_mrc = Mrc(map_file, apix=apix)
original_size = map_mrc.data.shape[0]
print('finish reading the volume')

star = RelionStar(star_file)
phi, theta, psi, x, y = read_relion_transformation(star)
du, dv, da, cs, kv, ac = read_relion_ctf(star)
if cs is None:
    cs = 0.0
else:
    cs = cs[0]
if kv is None:
    kv = 200
else:
    kv = kv[0]
if ac is None:
    ac = 0.0
else:
    ac = ac[0]
print('finish reading star file')

f_3d = preprocess_for_projection(map_mrc.data, pad, trilinear=trilinear)

print('finish fourier transform of 3D map')

stack_data = np.empty((len(x), original_size, original_size), dtype=np.float32)

for i in range(len(x)):
    f_2d = np.zeros((original_size, original_size // 2 + 1), dtype=np.complex)

    projection(f_2d, f_3d,
               phi[i], theta[i], psi[i], -x[i], -y[i],
               pad,
               True,
               du[i], dv[i], da[i], cs, apix=apix, kv=kv, ac=ac, white=white,
               trilinear=trilinear)

    stack_data[i, :, :] = np.fft.irfft2(f_2d)
    print('finish projecting: particle ', i)

stack_mrc = Mrc(stack_data, apix=apix)
stack_mrc.write(stack_file)



