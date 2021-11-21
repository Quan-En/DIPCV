import numpy as np
from numpy import array
from math import pi

from utils import blockshaped

def DFT(image:np.array):
    nrows, ncols = image.shape[0], image.shape[1]
    all_row_index, all_col_index = np.indices((nrows, ncols))

    norm_all_row_index = all_row_index / nrows
    norm_all_col_index = all_col_index / ncols

    index_array = np.kron(all_row_index, norm_all_row_index) + np.kron(all_col_index, norm_all_col_index)
    weighted = np.exp(-array([1j]) * 2 * pi * index_array)
    weighted_array_list = blockshaped(weighted, nrows, ncols)
    result = list(map(lambda x: np.sum(x*image), weighted_array_list))
    return array(result).reshape(nrows, ncols)


def IDFT(freq_map):
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]
    freq_map = freq_map / (nrows*ncols)
    all_row_index, all_col_index = np.indices((nrows, ncols))

    norm_all_row_index = all_row_index / nrows
    norm_all_col_index = all_col_index / ncols
    index_array = np.kron(all_row_index, norm_all_row_index) + np.kron(all_col_index, norm_all_col_index)
    weighted = np.exp(array([1j]) * 2 * pi * index_array)
    weighted_array_list = blockshaped(weighted, nrows, ncols)
    result = list(map(lambda x: np.sum(x*freq_map), weighted_array_list))
    return array(result).reshape(nrows, ncols)