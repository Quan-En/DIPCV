import numpy as np
from numpy import pi
from utils import create_image_radius_index

## Low-pass
def ideal_low_pass(freq_map, D0=10):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    filter_index = np.sqrt(all_row_index**2 + all_col_index**2) <= D0
    output = np.zeros(freq_map.shape)
    output[filter_index] = 1
    return freq_map * output

def gaussian_low_pass(freq_map, sigma=100):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    output = np.exp(-(all_row_index**2 + all_col_index**2)/(2*(sigma**2)))
    return freq_map * output

def butterworth_low_pass(freq_map, D0=100, n=1):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    D_index = np.sqrt(all_row_index**2 + all_col_index**2)
    output = 1 / (1 + (D_index / D0)**(2*n))
    return freq_map * output

## High-pass
def ideal_high_pass(freq_map, D0=10):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    filter_index = np.sqrt(all_row_index**2 + all_col_index**2) <= D0
    output = np.zeros(freq_map.shape)
    output[filter_index] = 1
    return freq_map * (1-output)

def butterworth_high_pass(freq_map, D0=10, n=1):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    D_index = np.sqrt(all_row_index**2 + all_col_index**2)
    output = 1 / (1 + (D_index / D0)**(2*n))
    return freq_map * (1-output)

## Gaussian filter
def gaussian_pass(freq_map, sigma=10):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    output_left = 1 / (2 * pi * (sigma**2))
    output_right =  np.exp(-(all_row_index**2 + all_col_index**2)/(2*(sigma**2)))
    output = output_left * output_right
    return freq_map * output

## Inverse filter
def inverse_pass(freq_map, sigma=10):
    # use gaussian filter
    output = gaussian_pass(freq_map, sigma)
    return freq_map * (1 / output)

## Wiener filter
def wiener_pass(freq_map, sigma=10, k=10):
    output = gaussian_pass(freq_map, sigma)
    output = np.conj(output) / (np.abs(output)**2 + k)
    return freq_map * output
