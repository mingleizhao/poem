# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 21:18:38 2016

author: Minglei Zhao
contact: mingleizhao@gmail.com

This module exports classes and functions for star files:
    Class:
        RelionStar(file_name | data dictionary):
            star files output by Relion

"""


import collections
import numpy as np


class RelionStar:
    """
    Contain data and basic methods for a Relion star file.

    Public variables
    ----------------
    self.data_dict: collections.OrderedDict(str : collections.OrderedDict(
                   str : int | float | str | numpy.ndarray))
        The main dictionary containing all the data in a star file. First
        string is referred to as block name. Second string is referred to as key
        name.
    self.block_name_list: list
        Contain a list of all the block names.
    self.key_name_list: list(list)
        Contain a list of all the key names grouped based on the block names.
    self.type: str
        Contain the type of the star file.
    self.file: str
        Contain the path to the source file. None if created from data.

    Public methods
    --------------
    self.find(self, search_string: str) -> Tuple
    self.write(self, file_name: str)

    """

    def __init__(self, *args):
        """
        Initialize the class object with either a star file or a dictionary.

        Parameters
        ----------
        *args: str | dict
            A string containing path to a star file or a dictionary with
            specific structure:
                collections.OrderedDict(str : collections.OrderedDict(
                str : int | float | str | numpy.ndarray))
            An empty object is returned otherwise.

        Examples
        --------
        >>> relion_star_object = RelionStar('path/some_relion_output.star')

        """
        # instance variables
        self.data_dict = collections.OrderedDict()
        self.block_name_list = []
        self.key_name_list = []
        self.type = None
        self.file = None
        # parse the arguments
        if len(args) != 1:
            print('RelionStar: empty object created!')
        elif isinstance(args[0], str):
            self._initialize_from_file(args[0])
            self._determine_star_file_type()
        elif isinstance(args[0], dict):
            self._initialize_from_dictionary(args[0])
            self._determine_star_file_type()
        else:
            raise TypeError('RelionStar: only string or dictionary argument!')

    def _initialize_from_file(self, star_file):
        f = open(star_file, 'r')
        self.file = star_file
        # parse relion star file
        temp_dict = collections.OrderedDict()
        temp_list = []
        block_name = ''
        is_in_loop = False
        is_parsing = False
        for line in f:
            stripped_line = line.strip()
            if stripped_line.startswith('data'):
                block_name = stripped_line
                self.block_name_list.append(block_name)
                self.key_name_list.append([])
                key_name_list_index = len(self.key_name_list) - 1
            elif stripped_line == 'loop_':
                is_in_loop = True
            elif stripped_line.startswith('_'):
                is_parsing = True
                if is_in_loop:
                    temp_dict[line.strip()] = []
                    temp_list.append(stripped_line)
                    self.key_name_list[key_name_list_index].append(
                        stripped_line)
                else:
                    data_line = line.split()
                    temp_dict[data_line[0]] = _convert_str_to_number(
                        data_line[1])
                    self.key_name_list[key_name_list_index].append(
                        data_line[0])
            elif stripped_line == '':
                if is_parsing:
                    if is_in_loop:
                        for keys in temp_dict:
                            temp_dict[keys] = np.array(temp_dict[keys])
                    self.data_dict[block_name] = temp_dict
                    is_parsing = False
                    is_in_loop = False
                    temp_dict = collections.OrderedDict()
                    temp_list = []
                else:
                    pass  # ignore blank line
            elif stripped_line.startswith('#'):
                pass  # ignore comments
            else:
                data_line = line.split()
                for i in range(len(data_line)):
                    temp_dict[temp_list[i]].append(_convert_str_to_number(
                                                 data_line[i]))

        # end of parsing in case the last line is not empty
        if stripped_line != '' and is_parsing and is_in_loop:
            for keys in temp_dict:
                temp_dict[keys] = np.array(temp_dict[keys])
            self.data_dict[block_name] = temp_dict

        f.close()

    def _initialize_from_dictionary(self, relion_dict):
        self.star_file = None
        self.data_dict = relion_dict
        self.block_name_list = list(relion_dict.keys())
        for keys in relion_dict:
            self.key_name_list.append(list(relion_dict[keys].keys()))

    def _determine_star_file_type(self):
        if 'data_' in self.block_name_list:
            self.type = 'data'
        elif 'data_images' in self.block_name_list:
            self.type = 'image'
        elif 'data_model_general' in self.block_name_list:
            self.type = 'model'
        elif 'data_general' in self.block_name_list:
            self.type = 'final_fsc'
        elif 'data_optimiser_general' in self.block_name_list:
            self.type = 'optimiser'
        elif 'data_sampling_general' in self.block_name_list:
            self.type = 'sampling'
        else:
            self.type = 'other'

    def find(self, search_string):
        """
        Return a tuple with block and key names matching the search string.

        Parameters
        ----------
        search_string: str

        Returns
        -------
        (block_name, key_name): tuple(str, str)
            if search string is a key name
        (block_name, ''): tuple(str, str)
            if search string is a block name
        (None, None)
            if no hit
        (block_name, key_name, block_name, key_name, ...)
            if multiple hits

        Note
        ----
        use "block_name | key_name" for specific search

        Examples
        --------
        >>> relion_star_object = RelionStar('star_file')
        >>> relion_star_object.find('ReferenceImage')
        ('data_model_classes', '_rlnReferenceImage #1')
        >>> relion_star_object.find('rlnResolution')
        RelionStar: Multiple hits with the search string!
        ('data_fsc', '_rlnResolution #2', 'data_guinier',
         '_rlnResolutionSquared #1')
        >>> relion_star_object.find('data_fsc | rlnResolution')
        ('data_fsc', '_rlnResolution #2')

        """
        block_string = ''
        pos = search_string.find('|')
        if pos != -1:
            block_string = search_string[:pos].strip()
            search_string = search_string[pos+1:].strip()
        found = []
        number_of_hit = 0
        for i in range(len(self.block_name_list)):
            if search_string in self.block_name_list[i]:
                number_of_hit += 1
                found.append(self.block_name_list[i])
                found.append('')
                continue
            for j in self.key_name_list[i]:
                if search_string in j:
                    found.append(self.block_name_list[i])
                    found.append(j)
                    number_of_hit += 1
        if number_of_hit == 0:
            print('RelionStar: No hit found!', end='\n')
            return None, None
        elif number_of_hit > 1:
            if block_string != '':
                temp_list = []
                for i in range(0, len(found), 2):
                    if block_string in found[i]:
                        temp_list.append(found[i])
                        temp_list.append(found[i+1])
                found = temp_list
            if len(found) > 2:
                print('RelionStar: Multiple hits with the search string!',
                      end='\n')
                return tuple(found)
            elif len(found) == 2:
                return tuple(found)
            else:
                print('RelionStar: No hit found!', end='\n')
                return None, None
        else:
            return tuple(found)

    def write(self, file_name):
        """
        Write a star file with current self.data_dict

        Parameters
        ----------
        file_name: str
            path and name of the new file

        """
        f = open(file_name, 'w')
        for block_name in self.block_name_list:
            f.write(block_name + '\n\n')
            is_in_loop = False
            for key_name in self.data_dict[block_name].keys():
                if ((isinstance(self.data_dict[block_name][key_name],
                                np.ndarray) or
                     isinstance(self.data_dict[block_name][key_name],
                                list)) and
                   (not is_in_loop)):
                    f.write('loop_' + '\n')
                    is_in_loop = True
                    f.write(key_name + '\n')
                elif (isinstance(self.data_dict[block_name][key_name],
                                 np.ndarray) or
                      isinstance(self.data_dict[block_name][key_name], list)):
                    f.write(key_name + '\n')
                else:
                    f.write('{0:40s} {1:12}\n'.format(key_name,
                            self.data_dict[block_name][key_name]))
            if is_in_loop:
                for i in range(len(self.data_dict[block_name][key_name])):
                    for key_name in self.data_dict[block_name].keys():
                        f.write('{0:12} '.format(
                            self.data_dict[block_name][key_name][i]))
                    f.write('\n')
            f.write('\n\n')
        f.close()


def _convert_str_to_number(value):
    # helper function
    # convert string to integer or float number
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_relion_transformation(star):
    block, key = star.find('data | AngleRot')
    phi = star.data_dict[block][key]
    block, key = star.find('data | AngleTilt')
    theta = star.data_dict[block][key]
    block, key = star.find('data | AnglePsi')
    psi = star.data_dict[block][key]
    block, key = star.find('data | OriginX')
    x = star.data_dict[block][key]
    block, key = star.find('data | OriginY')
    y = star.data_dict[block][key]
    return phi, theta, psi, x, y


def read_relion_ctf(star):
    # find defocus, no default, must find successfully
    block, key = star.find('data | DefocusU')
    defocus_u = star.data_dict[block][key]

    block, key = star.find('data | DefocusV')
    defocus_v = star.data_dict[block][key]

    block, key = star.find('data | DefocusAngle')
    defocus_a = star.data_dict[block][key]

    # find spherical aberration
    block, key = star.find('data | SphericalAberration')
    if block is not None and key is not None:
        cs = star.data_dict[block][key]
    else:
        cs = None

    # find voltage
    block, key = star.find('data | Voltage')
    if block is not None and key is not None:
        kv = star.data_dict[block][key]
    else:
        kv = None

    # find amplitude contrast
    block, key = star.find('data | AmplitudeContrast')
    if block is not None and key is not None:
        ac = star.data_dict[block][key]
    else:
        ac = None

    return defocus_u, defocus_v, defocus_a, cs, kv, ac

