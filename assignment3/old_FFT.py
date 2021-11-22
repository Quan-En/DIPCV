import numpy as np
from numpy import array
from math import pi

from DFT import DFT, IDFT


def FFT(image):
    image = np.asarray(image, dtype=float)
    nrows, ncols = image.shape[-2], image.shape[-1]
    
    if (nrows % 2 > 0) or (nrows != ncols):
        raise ValueError("size of x must be a power of 2 and (height, width) should equal.")
    elif nrows <= 32:
        return DFT(image)
    else:
        row_even_col_even = FFT(image[::2,::2])
        row_odd_col_even = FFT(image[1::2,::2])
        row_even_col_odd = FFT(image[::2,1::2])
        row_odd_col_odd = FFT(image[1::2,1::2])
        
        row_factor = np.exp(-2j * np.pi * np.arange(nrows) / nrows)
        col_factor = np.exp(-2j * np.pi * np.arange(ncols) / ncols)
        row_factor_matrix = np.repeat(row_factor.reshape(1,-1),repeats=nrows, axis=0)
        col_factor_matrix = np.repeat(col_factor.reshape(-1,1),repeats=ncols, axis=1)

        # concatenate block matrix

        block_11 = row_even_col_even + \
            row_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * row_odd_col_even + \
            col_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * row_even_col_odd + \
            row_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * col_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * row_odd_col_odd

        block_12 = row_even_col_even + \
            row_factor_matrix[:int(nrows / 2),int(nrows / 2):] * row_odd_col_even + \
            col_factor_matrix[:int(nrows / 2),int(nrows / 2):] * row_even_col_odd + \
            row_factor_matrix[:int(nrows / 2),int(nrows / 2):] * col_factor_matrix[:int(nrows / 2),int(nrows / 2):] * row_odd_col_odd

        block_21 = row_even_col_even + \
            row_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * row_odd_col_even + \
            col_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * row_even_col_odd + \
            row_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * col_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * row_odd_col_odd


        block_22 = row_even_col_even + \
            row_factor_matrix[int(nrows / 2):,int(nrows / 2):] * row_odd_col_even + \
            col_factor_matrix[int(nrows / 2):,int(nrows / 2):] * row_even_col_odd + \
            row_factor_matrix[int(nrows / 2):,int(nrows / 2):] * col_factor_matrix[int(nrows / 2):,int(nrows / 2):] * row_odd_col_odd
        
        return np.block([
            [block_11, block_12],
            [block_21, block_22]
        ])


def IFFT(image):
    nrows, ncols = image.shape[-2], image.shape[-1]
    
    if (nrows % 2 > 0) or (nrows != ncols):
        raise ValueError("size of x must be a power of 2 and (height, width) should equal.")
    elif nrows <= 32:
        return IDFT(image)
    else:
        row_even_col_even = IFFT(image[::2,::2])
        row_odd_col_even = IFFT(image[1::2,::2])
        row_even_col_odd = IFFT(image[::2,1::2])
        row_odd_col_odd = IFFT(image[1::2,1::2])
        
        row_factor = np.exp(2j * np.pi * np.arange(nrows) / nrows)
        col_factor = np.exp(2j * np.pi * np.arange(ncols) / ncols)
        row_factor_matrix = np.repeat(row_factor.reshape(1,-1),repeats=nrows, axis=0)
        col_factor_matrix = np.repeat(col_factor.reshape(-1,1),repeats=ncols, axis=1)

        # concatenate block matrix

        block_11 = row_even_col_even + \
            row_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * row_odd_col_even + \
            col_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * row_even_col_odd + \
            row_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * col_factor_matrix[:int(nrows / 2),:int(nrows / 2)] * row_odd_col_odd

        block_12 = row_even_col_even + \
            row_factor_matrix[:int(nrows / 2),int(nrows / 2):] * row_odd_col_even + \
            col_factor_matrix[:int(nrows / 2),int(nrows / 2):] * row_even_col_odd + \
            row_factor_matrix[:int(nrows / 2),int(nrows / 2):] * col_factor_matrix[:int(nrows / 2),int(nrows / 2):] * row_odd_col_odd

        block_21 = row_even_col_even + \
            row_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * row_odd_col_even + \
            col_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * row_even_col_odd + \
            row_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * col_factor_matrix[int(nrows / 2):,:int(nrows / 2)] * row_odd_col_odd


        block_22 = row_even_col_even + \
            row_factor_matrix[int(nrows / 2):,int(nrows / 2):] * row_odd_col_even + \
            col_factor_matrix[int(nrows / 2):,int(nrows / 2):] * row_even_col_odd + \
            row_factor_matrix[int(nrows / 2):,int(nrows / 2):] * col_factor_matrix[int(nrows / 2):,int(nrows / 2):] * row_odd_col_odd
        
        return np.block([
            [block_11, block_12],
            [block_21, block_22]
        ])