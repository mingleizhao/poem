# -*- coding: utf-8 -*-

"""
Created on Mon June 16 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

"""


import numpy as np
from poem.data.mrc import Mrc
from poem.data.star import RelionStar
from poem.data.star import read_relion_transformation
from poem.data.star import read_relion_ctf
from poem.em.back_projection import back_projection
from poem.em.back_projection import postprocess_for_reconstruction
from poem.em.check import sanity_check_stack


star_file = '/Users/minglei/PROJECTS/poem/test/70S.star'
map_file = '/Users/minglei/PROJECTS/poem/test/70S_reconstruct.mrc'
stack_file = '/Users/minglei/PROJECTS/poem/test/70S_projected.mrc'

pad = 2
res = 1.0
apix = 2.52
trilinear = True
white = True

stack = Mrc(stack_file, apix=apix)
sanity_check_stack(stack.data)
original_size = stack.data.shape[2]
padded_size = stack.data.shape[2] * pad
print('finish reading image stack')

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

f_3d = np.zeros((padded_size, padded_size, padded_size // 2 + 1),
                dtype=np.complex)

w_3d = np.zeros_like(f_3d, dtype=np.float)

w_2d = np.zeros((original_size, original_size // 2 + 1), dtype=np.float)

for i in range(len(x)):
    img = stack.data[i, :, :]

    f_2d = np.fft.rfft2(img)

    back_projection(f_2d, f_3d, w_2d, w_3d,
                    phi[i], theta[i], psi[i], x[i], y[i],
                    pad,
                    True,
                    du[i], dv[i], da[i], cs, apix=apix, kv=kv, ac=ac,
                    white=white, trilinear=trilinear)

    print('finish processing: particle ', i)

vol = postprocess_for_reconstruction(f_3d, w_3d, pad, trilinear)

output = Mrc(vol, apix=apix)
output.write(map_file)

