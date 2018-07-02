def projection_old(arr_2d, arr_3d, original_size,
               phi, theta, psi, x, y,
               pad=2,
               interpolator='trilinear'):
    # make a copy of the coordinates
    original_coords = np.copy(arr_2d['INDE'])

    # apply rotation to empty 2d array
    rotate_3d(arr_2d, phi, theta, psi)

    # interpolate padded 3d array to rotated empty 2d array
    interpolate(arr_2d, arr_3d, original_size, pad=pad,
                interpolator=interpolator)

    # use old coordinates for following steps
    arr_2d['INDE'] = original_coords

    # apply translation to 2d array
    translate_2d(arr_2d, original_size, x, y)


def preprocess_3d_before_projection_old(vol, pad=2, interpolator='trilinear'):
    # sanity check
    if vol.ndim != 3:
        raise ValueError('preprocess_3d: volume has to be a cubic!')
    if vol.shape[0] != vol.shape[1] or vol.shape[0] != vol.shape[2]:
        raise ValueError('preprocess_3d: volume has to be a cubic!')

    # assign sizes
    original_size = vol.shape[0]
    padded_size = original_size * pad
    half_padded_size = padded_size // 2

    # sinc correct
    if interpolator == 'nn':
        vol /= sinc_3d_center_full(original_size, pad)
    elif interpolator == 'trilinear':
        vol /= (sinc_3d_center_full(original_size, pad) ** 2)

    # padding
    vol_pad = pad_3d(vol, pad)

    # fourier transform
    f_3d = np.fft.rfftn(vol_pad)

    # center fourier transform
    f_3d_center = np.fft.fftshift(f_3d, axes=(0, 1))

    # convert to array
    arr_3d = vol_to_arr(f_3d_center)

    # center object for 3D transformation
    translate_3d(arr_3d, padded_size,
                 -half_padded_size,
                 -half_padded_size,
                 -half_padded_size)

    return arr_3d


def postprocess_2d_after_projection_old(arr_2d, size):
    # center object to the image
    translate_2d(arr_2d, size, size // 2, size // 2)

    # convert to image
    f_2d = arr_to_img(arr_2d, size)

    # center fourier transform
    f_2d_center = np.fft.fftshift(f_2d, axes=0)

    # fourier transform
    img = np.fft.irfft2(f_2d_center)

    return img.astype(np.float32)


def projection_old(f_2d, f_3d,
               phi, theta, psi, x, y,
               pad=2,
               interpolator='trilinear'):
    # interpolate padded 3d array to rotated empty 2d array
    interpolate(f_2d, f_3d, phi, theta, psi, pad, interpolator)

    # apply translation to 2d array
    translate_2d(f_2d, x, y)


def preprocess_3d_before_projection_old(vol, pad=2, interpolator='trilinear'):
    # sanity check
    if vol.ndim != 3:
        raise ValueError('preprocess_3d: volume has to be a cubic!')
    if vol.shape[0] != vol.shape[1] or vol.shape[0] != vol.shape[2]:
        raise ValueError('preprocess_3d: volume has to be a cubic!')

    # sinc correct
    if interpolator == 'trilinear':
        sinc_correction_3d(vol, pad, True)
    else:
        sinc_correction_3d(vol, pad, False)

    # padding
    vol_pad = pad_3d(vol, pad)

    # fourier transform
    f_3d = np.fft.rfftn(vol_pad)

    # center object for 3D transformation
    center_3d(f_3d)

    return f_3d


def postprocess_2d_after_projection(f_2d):
    # center object to the image
    center_2d(f_2d)

    # fourier transform
    img = np.fft.irfft2(f_2d)

    return img.astype(np.float32)


def v_interpolate(arr_2d, arr_3d, original_size,
                  pad=2, interpolator='trilinear'):
    padded_size = original_size * pad
    half_padded_size = padded_size // 2
    x = arr_2d['INDE'][:, 0] * pad
    y = arr_2d['INDE'][:, 1] * pad
    z = arr_2d['INDE'][:, 2] * pad

    # for negative x find Friedel's pair
    neg = x < 0
    x[neg] *= -1
    y[neg] *= -1
    z[neg] *= -1

    if interpolator == 'nn':
        x = np.round(x)
        y = np.round(y)
        z = np.round(z)
        mask = x ** 2 + y ** 2 + z ** 2 < half_padded_size ** 2
        arr_2d['F'][mask] = arr_3d['F'][v_find_indices_3d(x[mask],
                                                          y[mask],
                                                          z[mask],
                                                          padded_size
                                                          ).astype(int)]
        arr_2d['F'][np.logical_and(mask, neg)] = np.conj(
            arr_2d['F'][np.logical_and(mask, neg)])

    elif interpolator == 'trilinear':
        x0 = np.floor(x)
        x1 = x0 + 1
        xd = x - x0
        y0 = np.floor(y)
        y1 = y0 + 1
        yd = y - y0
        z0 = np.floor(z)
        z1 = z0 + 1
        zd = z - z0
        mask = np.max((np.abs(x0), np.abs(x1)), axis=0) ** 2 + \
               np.max((np.abs(y0), np.abs(y1)), axis=0) ** 2 + \
               np.max((np.abs(z0), np.abs(z1)), axis=0) ** 2 < \
               half_padded_size ** 2
        x0_m = x0[mask]
        x1_m = x1[mask]
        xd_m = xd[mask]
        y0_m = y0[mask]
        y1_m = y1[mask]
        yd_m = yd[mask]
        z0_m = z0[mask]
        z1_m = z1[mask]
        zd_m = zd[mask]
        d000 = arr_3d['F'][v_find_indices_3d(x0_m, y0_m, z0_m,
                                             padded_size).astype(int)]
        d100 = arr_3d['F'][v_find_indices_3d(x1_m, y0_m, z0_m,
                                             padded_size).astype(int)]
        d010 = arr_3d['F'][v_find_indices_3d(x0_m, y1_m, z0_m,
                                             padded_size).astype(int)]
        d001 = arr_3d['F'][v_find_indices_3d(x0_m, y0_m, z1_m,
                                             padded_size).astype(int)]
        d110 = arr_3d['F'][v_find_indices_3d(x1_m, y1_m, z0_m,
                                             padded_size).astype(int)]
        d101 = arr_3d['F'][v_find_indices_3d(x1_m, y0_m, z1_m,
                                             padded_size).astype(int)]
        d011 = arr_3d['F'][v_find_indices_3d(x0_m, y1_m, z1_m,
                                             padded_size).astype(int)]
        d111 = arr_3d['F'][v_find_indices_3d(x1_m, y1_m, z1_m,
                                             padded_size).astype(int)]
        d00 = d000 * (1 - xd_m) + d100 * xd_m
        d01 = d001 * (1 - xd_m) + d101 * xd_m
        d10 = d010 * (1 - xd_m) + d110 * xd_m
        d11 = d011 * (1 - xd_m) + d111 * xd_m
        d0 = d00 * (1 - yd_m) + d10 * yd_m
        d1 = d01 * (1 - yd_m) + d11 * yd_m
        arr_2d['F'][mask] = d0 * (1 - zd_m) + d1 * zd_m
        arr_2d['F'][np.logical_and(mask, neg)] = np.conj(
            arr_2d['F'][np.logical_and(mask, neg)])

    else:
        raise ValueError('interpolate: interpolator not defined!')


def interpolate_old(arr_2d, arr_3d, original_size, pad=2, interpolator='trilinear'):
    if interpolator == 'trilinear':
        interpolate_tri(arr_2d, arr_3d, original_size, pad)
    elif interpolator == 'nn':
        interpolate_nn(arr_2d, arr_3d, original_size, pad)
    else:
        raise ValueError('interpolate: interpolator not defined!')


@jit(nopython=True)
def interpolate_nn_old(arr_2d, arr_3d, original_size, pad):
    padded_size = original_size * pad
    half_padded_size = padded_size // 2
    for i in range(len(arr_2d)):
        x = arr_2d[i]['INDE'][0] * pad
        y = arr_2d[i]['INDE'][1] * pad
        z = arr_2d[i]['INDE'][2] * pad
        if x ** 2 + y ** 2 + z ** 2 < (half_padded_size - 1) ** 2:
            if x < 0:
                neg = True
                x = -x
                y = -y
                z = -z
            else:
                neg = False
            x = np.round(x)
            y = np.round(y)
            z = np.round(z)
            coord = int(v_find_indices_3d(x, y, z, padded_size))
            if neg:
                arr_2d[i]['F'] = np.conj(arr_3d[coord]['F'])
            else:
                arr_2d[i]['F'] = arr_3d[coord]['F']


@jit(nopython=True)
def interpolate_tri_old(arr_2d, arr_3d, original_size, pad):
    padded_size = original_size * pad
    half_padded_size = padded_size // 2
    for i in range(len(arr_2d)):
        x = arr_2d[i]['INDE'][0] * pad
        y = arr_2d[i]['INDE'][1] * pad
        z = arr_2d[i]['INDE'][2] * pad
        if x ** 2 + y ** 2 + z ** 2 < (half_padded_size - 1) ** 2:
            if x < 0:
                neg = True
                x = -x
                y = -y
                z = -z
            else:
                neg = False
            x0 = np.floor(x)
            x1 = x0 + 1
            xd = x - x0
            y0 = np.floor(y)
            y1 = y0 + 1
            yd = y - y0
            z0 = np.floor(z)
            z1 = z0 + 1
            zd = z - z0
            d000 = arr_3d[int(v_find_indices_3d(x0, y0, z0, padded_size))]['F']
            d100 = arr_3d[int(v_find_indices_3d(x1, y0, z0, padded_size))]['F']
            d010 = arr_3d[int(v_find_indices_3d(x0, y1, z0, padded_size))]['F']
            d001 = arr_3d[int(v_find_indices_3d(x0, y0, z1, padded_size))]['F']
            d110 = arr_3d[int(v_find_indices_3d(x1, y1, z0, padded_size))]['F']
            d101 = arr_3d[int(v_find_indices_3d(x1, y0, z1, padded_size))]['F']
            d011 = arr_3d[int(v_find_indices_3d(x0, y1, z1, padded_size))]['F']
            d111 = arr_3d[int(v_find_indices_3d(x1, y1, z1, padded_size))]['F']
            d00 = d000 * (1 - xd) + d100 * xd
            d01 = d001 * (1 - xd) + d101 * xd
            d10 = d010 * (1 - xd) + d110 * xd
            d11 = d011 * (1 - xd) + d111 * xd
            d0 = d00 * (1 - yd) + d10 * yd
            d1 = d01 * (1 - yd) + d11 * yd
            if neg:
                arr_2d[i]['F'] = np.conj(d0 * (1 - zd) + d1 * zd)
            else:
                arr_2d[i]['F'] = d0 * (1 - zd) + d1 * zd


def inv_interpolate(arr_2d, arr_3d, w_2d, w_3d, original_size,
                    pad=2, interpolator='trilinear'):
    if w_2d is not None and len(arr_2d) != len(w_2d):
        raise ValueError('inv_interpolate: weight array must be the same '
                         'size as input array!')
    padded_size = original_size * pad
    half_padded_size = padded_size // 2
    x = arr_2d['INDE'][:, 0] * pad
    y = arr_2d['INDE'][:, 1] * pad
    z = arr_2d['INDE'][:, 2] * pad
    v = arr_2d['F']

    # for negative x find Friedel's pair
    neg = x < 0
    x[neg] *= -1
    y[neg] *= -1
    z[neg] *= -1
    v[neg] = np.conj(v[neg])

    if interpolator == 'nn':
        x = np.round(x)
        y = np.round(y)
        z = np.round(z)
        mask = x ** 2 + y ** 2 + z ** 2 < half_padded_size ** 2
        ind = v_find_indices_3d(x[mask], y[mask], z[mask],
                                padded_size).astype(int)
        arr_3d['F'][ind] += v[mask]
        if w_2d is None:
            w_3d['F'][ind] += 1.0
        else:
            w_3d['F'][ind] += w_2d[mask]

    elif interpolator == 'trilinear':
        x0 = np.floor(x)
        x1 = x0 + 1
        xd = x - x0
        y0 = np.floor(y)
        y1 = y0 + 1
        yd = y - y0
        z0 = np.floor(z)
        z1 = z0 + 1
        zd = z - z0
        mask = np.max((np.abs(x0), np.abs(x1)), axis=0) ** 2 + \
               np.max((np.abs(y0), np.abs(y1)), axis=0) ** 2 + \
               np.max((np.abs(z0), np.abs(z1)), axis=0) ** 2 < \
               half_padded_size ** 2
        x0_m = x0[mask]
        x1_m = x1[mask]
        xd_m = xd[mask]
        y0_m = y0[mask]
        y1_m = y1[mask]
        yd_m = yd[mask]
        z0_m = z0[mask]
        z1_m = z1[mask]
        zd_m = zd[mask]

        # inverse trilinear
        d0 = 1 - zd_m
        d1 = zd_m
        d00 = d0 * (1 - yd_m)
        d01 = d1 * (1 - yd_m)
        d10 = d0 * yd_m
        d11 = d1 * yd_m
        d000 = d00 * (1 - xd_m)
        d100 = d00 * xd_m
        d010 = d10 * (1 - xd_m)
        d001 = d01 * (1 - xd_m)
        d110 = d10 * xd_m
        d101 = d01 * xd_m
        d011 = d11 * (1 - xd_m)
        d111 = d11 * xd_m
        ind000 = v_find_indices_3d(x0_m, y0_m, z0_m, padded_size)
        ind100 = v_find_indices_3d(x1_m, y0_m, z0_m, padded_size)
        ind010 = v_find_indices_3d(x0_m, y1_m, z0_m, padded_size)
        ind001 = v_find_indices_3d(x0_m, y0_m, z1_m, padded_size)
        ind110 = v_find_indices_3d(x1_m, y1_m, z0_m, padded_size)
        ind101 = v_find_indices_3d(x1_m, y0_m, z1_m, padded_size)
        ind011 = v_find_indices_3d(x0_m, y1_m, z1_m, padded_size)
        ind111 = v_find_indices_3d(x1_m, y1_m, z1_m, padded_size)
        arr_3d['F'][ind000] += d000 * v[mask]
        arr_3d['F'][ind100] += d100 * v[mask]
        arr_3d['F'][ind010] += d010 * v[mask]
        arr_3d['F'][ind001] += d001 * v[mask]
        arr_3d['F'][ind110] += d110 * v[mask]
        arr_3d['F'][ind101] += d101 * v[mask]
        arr_3d['F'][ind011] += d011 * v[mask]
        arr_3d['F'][ind111] += d111 * v[mask]
        if w_2d is None:
            w_3d['F'][ind000] += d000
            w_3d['F'][ind100] += d100
            w_3d['F'][ind010] += d010
            w_3d['F'][ind001] += d001
            w_3d['F'][ind110] += d110
            w_3d['F'][ind101] += d101
            w_3d['F'][ind011] += d011
            w_3d['F'][ind111] += d111
        else:
            w_3d['F'][ind000] += d000 * w_2d[mask]
            w_3d['F'][ind100] += d100 * w_2d[mask]
            w_3d['F'][ind010] += d010 * w_2d[mask]
            w_3d['F'][ind001] += d001 * w_2d[mask]
            w_3d['F'][ind110] += d110 * w_2d[mask]
            w_3d['F'][ind101] += d101 * w_2d[mask]
            w_3d['F'][ind011] += d011 * w_2d[mask]
            w_3d['F'][ind111] += d111 * w_2d[mask]
    else:
        raise ValueError('v_inv_interpolate: interpolator not defined!')


def empty_arr_2d(size):
    length = size * (size // 2 + 1)
    arr = np.zeros(length, dtype=[('INDE', np.float, 3), ('F', np.complex)])
    arr['INDE'] = indices_2d(size)  # fft-center axes and half mode
    return arr


def empty_arr_3d(size):
    length = size * size * (size // 2 + 1)
    arr = np.zeros(length, dtype=[('INDE', np.float, 3), ('F', np.complex)])
    arr['INDE'] = indices_3d(size)  # fft-center axes and half mode
    return arr


def empty_arr_3d_new(size):
    length = size * size * (size // 2 + 1)
    arr = np.empty(length, dtype=[('INDE', np.float, 3), ('F', np.complex)])
    load_arr(arr, size)
    return arr


@jit(nopython=True)
def load_arr(arr, size):
    half = size // 2
    for i in range(len(arr)):
        arr[i]['INDE'][0] = i % (half + 1)
        arr[i]['INDE'][1] = i // (half + 1) % size - half
        arr[i]['INDE'][2] = i // ((half + 1) * size) % size - half
        arr[i]['F'] = 0


def img_to_arr(img):
    # sanity check
    if len(img.shape) != 2:
        raise ValueError('img_to_arr: image dimension must be 2!')
    size = img.shape[0]
    half_size = size // 2
    if (half_size + 1) != img.shape[1]:
        raise ValueError('img_to_arr: image size not right!')

    # initialize structured array
    arr = empty_arr_2d(size)
    arr['F'] = img.flatten()
    return arr


def vol_to_arr(vol):
    # sanity check
    if len(vol.shape) != 3:
        raise ValueError('vol_to_arr: volume dimension must be 3!')
    size = vol.shape[0]
    half_size = size // 2
    if (half_size + 1) != vol.shape[2] or size != vol.shape[1]:
        raise ValueError('vol_to_arr: volume size not right for "half" mode!')

    # initialize structured array
    arr = empty_arr_3d(size)
    arr['F'] = vol.flatten()
    return arr


def arr_to_img(arr, size):
    length = size * (size // 2 + 1)
    if len(arr) < length:  # in the case that arr has been masked
        img_index = empty_arr_2d(size)
        img_index['F'][v_find_indices_2d(arr['INDE'][:, 0],
                                         arr['INDE'][:, 1],
                                         size).astype(int)] = arr['F']
    elif len(arr) == length:
        img_index = arr
    else:
        raise ValueError('arr_to_img: array size not correct!')

    return img_index['F'].reshape((size, size // 2 + 1))


def arr_to_vol(arr, size):
    length = size * size * (size // 2 + 1)
    if len(arr) < length:  # in the case that arr has been masked
        vol_index = empty_arr_3d(size)
        vol_index['F'][v_find_indices_3d(arr['INDE'][:, 0],
                                         arr['INDE'][:, 1],
                                         arr['INDE'][:, 2],
                                         size).astype(int)] = arr['F']
    elif len(arr) == length:
        vol_index = arr
    else:
        raise ValueError('arr_to_vol: array size not correct!')

    return vol_index['F'].reshape((size, size, size // 2 + 1))


@vectorize
def v_find_indices_2d(x, y, size):
    return (y + size // 2) * (size // 2 + 1) + x


@vectorize
def v_find_indices_3d(x, y, z, size):
    return (z + size // 2) * size * (size // 2 + 1) + \
           (y + size // 2) * (size // 2 + 1) + \
           x


def apply_ctf_2d_arr(arr_2d,
                     size,
                     defocus_u,  # unit: Angstrom
                     defocus_v,  # unit: Angstrom
                     defocus_a,  # in degree
                     cs,  # unit: mm
                     apix=1.0,  # unit: Angstrom/pixel
                     kv=200,  # unit: kilo voltage
                     ac=0.1,  # arbitrary unit
                     b_fac=0.0,
                     white=True):
    ctf = ctf_2d_arr(arr_2d['INDE'][:, 0],
                     arr_2d['INDE'][:, 1],
                     size,
                     defocus_u,  # unit: Angstrom
                     defocus_v,  # unit: Angstrom
                     defocus_a,  # in degree
                     cs,  # unit: mm
                     apix,  # unit: Angstrom/pixel
                     kv,  # unit: kilo voltage
                     ac,  # arbitrary unit
                     b_fac)
    if white:
        arr_2d['F'] *= ctf
    else:
        arr_2d['F'] *= -ctf
