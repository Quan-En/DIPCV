import numpy as np
from numpy import pi

from utils.Conv_Filter import Conv_Filter
from utils.utils import create_image_radius_index, freq_padding
from utils.FFT import FFT2, IFFT2

## Low-pass
def ideal_low_pass(freq_map, D0=10):
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]
    all_row_index, all_col_index = create_image_radius_index((nrows, ncols))
    filter_index = np.sqrt(all_row_index**2 + all_col_index**2) <= D0
    output = np.zeros(freq_map.shape)
    output[filter_index] = 1
    return freq_map * output

def gaussian_low_pass(freq_map, sigma=100):
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]
    all_row_index, all_col_index = create_image_radius_index((nrows, ncols))
    output = np.exp(-(all_row_index**2 + all_col_index**2)/(2*(sigma**2)))
    return freq_map * output

def butterworth_low_pass(freq_map, D0=100, n=1):
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]
    all_row_index, all_col_index = create_image_radius_index((nrows, ncols))
    D_index = np.sqrt(all_row_index**2 + all_col_index**2)
    output = 1 / (1 + (D_index / D0)**(2*n))
    return freq_map * output

## High-pass
def ideal_high_pass(freq_map, D0=10):
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]
    all_row_index, all_col_index = create_image_radius_index((nrows, ncols))
    filter_index = np.sqrt(all_row_index**2 + all_col_index**2) <= D0
    output = np.zeros(freq_map.shape)
    output[filter_index] = 1
    return freq_map * (1-output)

def butterworth_high_pass(freq_map, D0=10, n=1):
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]
    all_row_index, all_col_index = create_image_radius_index((nrows, ncols))
    D_index = np.sqrt(all_row_index**2 + all_col_index**2)
    output = 1 / (1 + (D_index / D0)**(2*n))
    return freq_map * (1-output)


## Resotration
conv_f = Conv_Filter()

### Gaussian filter
def gaussian_pass_filter(freq_map, win_size=3, sigma=1):
    # since `freq_map` is padding result
    # so we also padding the filter
    nrows, ncols = freq_map.shape[0], freq_map.shape[1]

    kernel = conv_f.gen_gaussian_kernel(win_size, sigma)
    freq_kernel = freq_padding(kernel, new_size=(nrows, ncols))
    output = FFT2(freq_kernel)
    return output

### Inverse filter
def inverse_pass(freq_map, win_size=3, sigma=1):
    # use gaussian filter
    output = gaussian_pass_filter(freq_map, win_size, sigma)
    return freq_map / (output+1e-6)

## Wiener filter
def wiener_pass(freq_map, win_size=3, sigma=1, k=1):
    output = gaussian_pass_filter(freq_map, win_size, sigma)
    output = np.conj(output) / (np.abs(output)**2 + k)
    return freq_map * output
