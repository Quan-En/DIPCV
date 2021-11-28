import numpy as np

from utils.Conv_Filter import Conv_Filter
from utils.utils import create_image_radius_index, freq_padding
from utils.FFT import FFT2

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

### (1) Inverse filter
def inverse_filter(freq_map, blur_kernel):
    # freq_map: frequency map
    # kernel: image is degraded by what kernel
    kernel_freq_map = FFT2(freq_padding(blur_kernel, new_size=freq_map.shape))
    return freq_map / (kernel_freq_map+1e-7) # add 1e-7 to avoid `divide zero`

### (2) Wiener filter
def wiener_filter(freq_map, blur_kernel, k=1):
    kernel_freq_map = FFT2(freq_padding(blur_kernel, new_size=freq_map.shape))
    kernel_freq_map = np.conj(kernel_freq_map) / (np.abs(kernel_freq_map)**2 + k)
    return freq_map * kernel_freq_map

def padding(image, win_size):    
    height, width = image.shape
    radius = win_size // 2
    paddedImage = np.zeros((height + 2*radius, width + 2*radius))

    # store original image to center location
    paddedImage[radius:height + radius,radius:width + radius] = image

    # The next few lines creates a padded image that reflects the (inner border)
    paddedImage[radius:height+radius, 0:radius] = np.fliplr(image[:,1:radius+1])
    paddedImage[radius:height+radius, width+radius:width+2*radius] = np.fliplr(image[:,width-radius-1:width-1])

    paddedImage[0:radius,:] = np.flipud(paddedImage[radius+1:2*radius+1,:])
    paddedImage[height+radius:height+2*radius,:] = np.flipud(paddedImage[height-1:height+radius-1,:])

    return paddedImage

### (3) Guided filter

def gen_averaging_kernel(win_size):
    g = np.ones((win_size, win_size))
    g = g / g.sum()
    return g

def mean_blur(image, win_size):
    
    image = padding(image, win_size).astype(float)
    height, width = image.shape

    # dst image height and width
    radius = win_size // 2
    dst_height = height - win_size + 1
    dst_width = width - win_size + 1
    
    result = np.zeros((dst_height, dst_width), dtype=np.float64)
    kernel = gen_averaging_kernel(win_size)

    for i in range(radius, radius+dst_height):
        for j in range(radius, radius+dst_width):
            result[i-radius, j-radius] = (image[i-radius:i+radius+1, j-radius:j+radius+1] * kernel).sum()

    return result

def guided_filter(image, guide_image, win_size, eps):
    # image: p
    # guide_image: I
    # output: q
    p = image.astype(float) / 255
    I = guide_image.astype(float) / 255

    mean_I = mean_blur(I, win_size)
    mean_p = mean_blur(p, win_size)

    corr_II = mean_blur(I*I, win_size)
    corr_Ip = mean_blur(I*p, win_size)

    var_I  = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps ** 2)
    b = mean_p - a * mean_I
    
    mean_a = mean_blur(a, win_size)
    mean_b = mean_blur(b, win_size)
    
    q = mean_a * I + mean_b
    return q * 255