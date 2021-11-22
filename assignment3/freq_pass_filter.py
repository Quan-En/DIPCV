import numpy as np
from utils import create_image_radius_index

def ideal_low_pass(freq_map, D0=10):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    filter_index = np.sqrt(all_row_index**2 + all_col_index**2) <= D0
    output = np.zeros(freq_map.shape)
    output[filter_index] = 1
    return output

def gaussian_low_pass(freq_map, sigma=100):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    output = np.exp(-(all_row_index**2 + all_col_index**2)/(2*(sigma**2)))
    return output

def butterworth_low_pass(freq_map, D0=100, n=1):
    all_row_index, all_col_index = create_image_radius_index(freq_map)
    D_index = np.sqrt(all_row_index**2 + all_col_index**2)
    output = 1 / (1 + (D_index / D0)**(2*n))
    return output


def ideal_high_pass(freq_map, D0=10):
    return 1 - ideal_low_pass(freq_map, D0)

def butterworth_high_pass(freq_map, D0=10, n=1):
    return 1 - butterworth_low_pass(freq_map, D0, n)
