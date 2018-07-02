# -*- coding: utf-8 -*-
"""
Created on June 5th 2017

author: Minglei Zhao
contact: mingleizhao@gmail.com

This module exports classes and functions for Mrc files:
    Class:
        Mrc(file_name | 3D numpy array, apix=):
            mrc format of cryo-EM data

"""

import sys
import numpy as np


class Mrc:
    """
    Contain data and basic I/O methods for Mrc file.

    Public variables
    ----------------
    Mrc.header_type: numpy.dtype
        Structured numpy array type for mrc header.
    self.header: numpy.ndarray
        A numpy structured array containing header information of the mrc file.
        The header is alway 1024 bytes.
    self.data: numpy.ndarray
        A numpy array containing all density values.
        Always a three-dimensional array.
        For 2D images, the third dimension is set to 1.
    self.apix: float
        Pixel size in angstrom. Can be initialized or updated.
    self.file: str
        Contain path to the source file. None if created from data.

    Public methods
    --------------
    self.printHeader(self)
    self.updateHeader(self)
    self.write(self, file_name: str)

    """
    #  define the header using numpy structured array
    #  'nc'               number of columns (fastest changing in map)
    #  'nr'               number of rows
    #  'ns'               number of sections (slowest changing in map)
    #  'mode'             mrc mode
    #  'ncstart'          number of first column in map (Default = 0)
    #  'nrstart'          number of first row in map
    #  'nsstart'          number of first section in map
    #  'nx'               number of intervals along X
    #  'ny'               number of intervals along Y
    #  'nz'               number of intervals along Z
    #  'x_length'         cell dimensions in angstroms
    #  'y_length'         cell dimensions in angstroms
    #  'z_length'         cell dimensions in angstroms
    #  'alpha'            cell angles in degrees (Default = 90)
    #  'betta'            cell angles in degrees (Default = 90)
    #  'gaama'            cell angles in degrees (Default = 90)
    #  'mapc'             axis corresp to cols (1,2,3 for X,Y,Z)
    #  'mapr'             axis corresp to rows (1,2,3 for X,Y,Z)
    #  'maps'             axis corresp to sections (1,2,3 for X,Y,Z)
    #  'amin'             minimum density value
    #  'amax'             maximum density value
    #  'amean'            mean density value
    #  'ispg'             space group number (default=1)
    #  'nsymbt'           number of bytes used for symmetry data
    #  'lskflg'
    #  'skwmat'           rotation matrix
    #  'skwtrn'           translation matrix
    #  'extra',           extra data, may include origin
    #  'mapi'             character string 'MAP' to identify file type
    #  'machst'           machine stamp (endianness)
    #  'rms'              rms deviation of map from mean density
    #  'nlabl'            number of labels being used
    #  'label_n'          10 80-character text labels
    header_type = np.dtype([('nc', np.int32, 1),
                            ('nr', np.int32, 1),
                            ('ns', np.int32, 1),
                            ('mode', np.int32, 1),
                            ('ncstart', np.int32, 1),
                            ('nrstart', np.int32, 1),
                            ('nsstart', np.int32, 1),
                            ('nx', np.int32, 1),
                            ('ny', np.int32, 1),
                            ('nz', np.int32, 1),
                            ('x_length', np.float32, 1),
                            ('y_length', np.float32, 1),
                            ('z_length', np.float32, 1),
                            ('alpha', np.float32, 1),
                            ('betta', np.float32, 1),
                            ('gaama', np.float32, 1),
                            ('mapc', np.int32, 1),
                            ('mapr', np.int32, 1),
                            ('maps', np.int32, 1),
                            ('amin', np.float32, 1),
                            ('amax', np.float32, 1),
                            ('amean', np.float32, 1),
                            ('ispg', np.int32, 1),
                            ('nsymbt', np.int32, 1),
                            ('lskflg', np.int32, 1),
                            ('skwmat', np.float32, 9),
                            ('skwtrn', np.float32, 3),
                            ('extra', np.int32, 15),
                            ('mapi', np.int8, 4),
                            ('machst', np.int8, 4),
                            ('rms', np.float32, 1),
                            ('nlabl', np.int32, 1),
                            ('label_n', np.int8, (10, 80))])

    def __init__(self, obj=None, apix=0.0):
        """
        Initialize the class object with either an mrc file or a numpy array.

        Parameters
        ----------
        *args: str | numpy.ndarray
            A string containing path to an mrc file or a three-dimensinal
            numpy.ndarray
            A default header is returned otherwise.

        Examples
        --------
        >>> object1 = Mrc('path/some_image.mrc')
        >>> object2 = Mrc(a_3D_numpy_array)

        """
        # parse the arguments
        if obj is None:
            print('Mrc: no data! header created!')
            self.apix = apix
            self.header = self._initialize_mrc_header()
            self.data = None
            self.file = None
        elif isinstance(obj, str):
            self.apix = apix
            self.file = obj
            self._initialize_from_file(obj)
        elif isinstance(obj, np.ndarray):
            self.apix = apix
            self.file = None
            self._initialize_from_ndarray(obj)
        else:
            raise TypeError('Mrc: string for filename or numpy.ndarray for ' +
                            'data')

    def _initialize_from_file(self, mrc_file):
        f = open(mrc_file, 'r')
        # read header
        self.header = np.fromfile(f, dtype=Mrc.header_type, count=1)
        # check endian-ness in the header
        data_endianness = 'little'  # default
        if (bytes(self.header['machst']) ==
           bytes(np.array([17, 17, 0, 0], dtype=np.int8))):
            data_endianness = 'big'
        if data_endianness != sys.byteorder:
            self.header.byteswap()
        # determine data type from mode
        if self.header['mode'] == 0:
            data_type = np.int8
        elif self.header['mode'] == 1:
            data_type = np.int16
        elif self.header['mode'] == 2:
            data_type = np.float32
        elif self.header['mode'] == 3:
            data_type = np.dtype((np.int16, 2))
        elif self.header['mode'] == 4:
            data_type = np.complex64
        else:
            raise ValueError('Mrc: unrecognized data mode!')
        # update apix
        if (self.header['x_length'] != 0 and self.header['y_length'] != 0 and
           self.header['z_length'] != 0):
            self.apix = float(self.header['x_length'] / self.header['nc'])
        # move file pointer to data
        f.seek(1024)
        temp_data = np.fromfile(f, dtype=data_type)
        if self.header['ns'][0] != 0:
            self.data = temp_data.reshape((self.header['ns'][0],
                                           self.header['nr'][0],
                                           self.header['nc'][0]))  # C type
        else:  # 2D image
            self.data = temp_data.reshape((self.header['nr'][0],
                                           self.header['nc'][0]))  # C type
        # swap data if endian-ness is different
        if data_endianness != sys.byteorder:
            self.data = self.data.byteswap()
        f.close()

    def _initialize_from_ndarray(self, ndarray):
        if ndarray.ndim != 2 and ndarray.ndim != 3:
            raise ValueError('Mrc: only take 2 or 3 dimensional data!')
        self.header = self._initialize_mrc_header()
        self.data = ndarray
        self.update_header()

    @staticmethod
    def _initialize_mrc_header():
        # assign default values to some header items
        header = np.zeros((1,), Mrc.header_type)
        header['alpha'] = 90
        header['betta'] = 90
        header['gaama'] = 90
        header['ispg'] = 1
        header['mapc'] = 1
        header['mapr'] = 2
        header['maps'] = 3
        header['mapi'] = np.fromstring(b'MAP ', dtype=np.int8)
        if sys.byteorder == 'little':  # little endian
            header['machst'] = np.array([68, 65, 0, 0])
        else:  # big endian
            header['machst'] = np.array([17, 17, 0, 0])
        return header

    def update_header(self):
        """
        Update header information based on current data array.

        Output
        ------
        Items that have been updated.

        Examples
        --------
        >>> mrc_object = Mrc('mrc_file')
        >>> mrc_object.update_header()
        Mrc: HEADER               NEW                  OLD
        Mrc: ispg                 [1]                  [0]
        Mrc: mapi                 MAP
        Mrc: machst               [[68 65  0  0]]      [[0 0 0 0]]
        Mrc: rms                  [ 13.83477783]       [ 0.]

        """
        if self.data is None:
            raise ValueError('Mrc: no data to update header!')
        new_header = self._initialize_mrc_header()
        # determine nc, nr, ns
        if self.data.ndim == 2:
            new_header['nc'] = self.data.shape[1]
            new_header['nr'] = self.data.shape[0]
            new_header['ns'] = 1
        else:
            new_header['nc'] = self.data.shape[2]
            new_header['nr'] = self.data.shape[1]
            new_header['ns'] = self.data.shape[0]
        # default mapc, mapr, maps
        # new_header['mapc'] = 1
        # new_header['mapr'] = 2
        # new_header['maps'] = 3
        # determine mode
        if self.data.dtype is np.dtype(np.int8):
            data_mode = 0
        elif self.data.dtype is np.dtype(np.int16):
            data_mode = 1
        elif self.data.dtype is np.dtype(np.float32):
            data_mode = 2
        elif self.data.dtype is np.dtype((np.int16, 2)):
            data_mode = 3
        elif self.data.dtype is np.dtype(np.complex64):
            data_mode = 4
        else:
            raise ValueError('Mrc: data type not recognized!')
        new_header['mode'] = data_mode
        # determine nx, ny, nz
        if self.data.shape[0] == self.data.shape[1] == self.data.shape[2]:
            new_header['nx'] = self.data.shape[2]
            new_header['ny'] = self.data.shape[1]
            new_header['nz'] = self.data.shape[0]
        else:  # a image stack or 2D image
            new_header['nx'] = self.data.shape[1]
            new_header['ny'] = self.data.shape[0]
            new_header['nz'] = 1
        # determine amin, amax, amean, rms
        new_header['amin'] = np.min(self.data)
        new_header['amax'] = np.max(self.data)
        new_header['amean'] = np.mean(self.data)
        new_header['rms'] = np.sqrt(np.sum(np.square(
                                   self.data - new_header['amean'])) /
                                   np.size(self.data))
        # determine x_length, y_length, z_length
        if self.apix != 0:
            if self.data.ndim == 2:
                new_header['x_length'] = self.apix * self.data.shape[1]
                new_header['y_length'] = self.apix * self.data.shape[0]
                new_header['z_length'] = 0
            else:
                new_header['x_length'] = self.apix * self.data.shape[2]
                new_header['y_length'] = self.apix * self.data.shape[1]
                new_header['z_length'] = self.apix * self.data.shape[0]
        # update nc, nr, ns, mode, nx, ny, nz,  x_length, y_length, z_length
        # alpha, betta, gaama, mapc, mapr, maps
        # amin, amax, amean, ispg, mapi, machst, rms
        updated_list = ['nc', 'nr', 'ns', 'mode', 'nx', 'ny', 'nz',
                        'x_length', 'y_length', 'z_length',
                        'alpha', 'betta', 'gaama',
                        'amin', 'amax', 'amean',
                        'ispg', 'mapi', 'machst', 'rms']
        first_line_printed = False
        for item in updated_list:
            if not np.allclose(new_header[item], self.header[item]):
                if not first_line_printed:
                    print('Mrc: {0:20s} {1:20s} {2:20s}'.
                          format('HEADER', 'NEW', 'OLD'))
                    first_line_printed = True
                if item == 'mapi':
                    print('Mrc: {0:20s} {1:20s} {2:20s}'.
                          format(item,
                                 ''.join([chr(ch)
                                          for ch in new_header[item][0]
                                          if chr(ch).isprintable()]),
                                 ''.join([chr(ch)
                                          for ch in self.header[item][0]
                                          if chr(ch).isprintable()])))
                else:
                    print('Mrc: {0:20s} {1:20s} {2:20s}'.
                          format(item, np.array_str(new_header[item]),
                                 np.array_str(self.header[item])))
                self.header[item] = new_header[item]

    def write(self, file_name):
        """
        Write an mrc file with current self.data

        Parameters
        ----------
        file_name: str
            path and name of the new file

        """
        self.update_header()
        with open(file_name, 'wb') as f:
            f.write(self.header.tostring()+self.data.tostring())

    def print_header(self):
        """
        Print header information.

        Note
        ----
        Items are printed not in the same order as in memory. Some
        non-printable items are omitted.

        Examples
        --------
        >>> mrc_object = Mrc('mrc_file')
        >>> mrc_object.print_header()
        Mrc: number of columns    [100]
        Mrc: number of rows       [100]
        Mrc: number of sections   [100]
        Mrc: data mode            [2]
        Mrc: first column         [0]
        Mrc: first row            [0]
        Mrc: first section        [0]
        Mrc: intervals along X    [100]
        Mrc: intervals along Y    [100]
        Mrc: intervals along Z    [100]
        Mrc: cell dimension X     [ 121.55999756]
        Mrc: cell dimension Y     [ 121.55999756]
        Mrc: cell dimension Z     [ 121.55999756]
        Mrc: cell angle alpha     [ 90.]
        Mrc: cell angle beta      [ 90.]
        Mrc: cell angle gaama     [ 90.]
        Mrc: space group          [1]
        Mrc: axis for column      [1]
        Mrc: axis for row         [2]
        Mrc: axis for section     [3]
        Mrc: minimal density      [-0.01463897]
        Mrc: maximal density      [ 0.03098994]
        Mrc: mean density         [ 0.00156577]
        Mrc: rms density          [ 0.0044822]
        Mrc: # extra labels       [0]

        """
        print('Mrc: {0:20s} {1:20s}'.
              format('number of columns', np.array_str(self.header['nc'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('number of rows', np.array_str(self.header['nr'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('number of sections', np.array_str(self.header['ns'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('data mode', np.array_str(self.header['mode'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('first column', np.array_str(self.header['ncstart'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('first row', np.array_str(self.header['nrstart'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('first section', np.array_str(self.header['nsstart'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('intervals along X', np.array_str(self.header['nx'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('intervals along Y', np.array_str(self.header['ny'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('intervals along Z', np.array_str(self.header['nz'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('cell dimension X',
                     np.array_str(self.header['x_length'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('cell dimension Y',
                     np.array_str(self.header['y_length'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('cell dimension Z',
                     np.array_str(self.header['z_length'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('cell angle alpha', np.array_str(self.header['alpha'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('cell angle beta', np.array_str(self.header['betta'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('cell angle gaama', np.array_str(self.header['gaama'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('space group', np.array_str(self.header['ispg'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('axis for column', np.array_str(self.header['mapc'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('axis for row', np.array_str(self.header['mapr'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('axis for section', np.array_str(self.header['maps'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('minimal density', np.array_str(self.header['amin'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('maximal density', np.array_str(self.header['amax'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('mean density', np.array_str(self.header['amean'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('rms density', np.array_str(self.header['rms'])))
        print('Mrc: {0:20s} {1:20s}'.
              format('# extra labels', np.array_str(self.header['nlabl'])))
        # print extra labels
        if 0 < self.header['nlabl'][0] <= 10:
            for i in range(self.header['nlabl'][0]):
                label = ''.join([chr(ch) for ch in self.header['label_n'][0][i]
                                 if chr(ch).isprintable()])
                print('Mrc: {0:20s} {1:80s}'.format('extra label', label))
