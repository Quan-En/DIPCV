# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:50:11 2021

@author: Taner
"""

# Packages
import os
# os.getcwd()
os.chdir("C:\\Users\\Taner\\Desktop\\master_degree\\lecture\\image_porcess_and_computer_vision\\HW\\HW2")
import cv2
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


# Read image file

raw_img_path_list = ["data\\F16.raw", "data\\flower.raw",
                     "data\\lena.raw", "data\\Noisy.raw", "data\\peppers.raw"]


img_array_list = []

for f_name in raw_img_path_list:
    img_array_list.append(np.fromfile(f_name, dtype=np.uint8).reshape(512, 512))

# Original figure
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    if i < 5:
        ax_i.imshow(img_array_list[i], cmap='gray')
fig.show()

# Original histogram plot
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    if i < 5:
        d = img_array_list[i].reshape(-1).astype(np.int)
        n, bins, patches = ax_i.hist(x=d, bins='auto', color='#0504aa', 
                                    alpha=0.7, rwidth=0.85)
        maxfreq = n.max()
        ax_i.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        ax_i.grid(axis='y', alpha=0.75)
        ax_i.set_xlabel('Value')
        ax_i.set_ylabel('Frequency')
        ax_i.set_title(r'$\mu = '+ str(round(np.mean(d),2)) + '$')
        # ax_i.imshow(img_array_list[i], cmap='gray')
fig.show()


# https://codeinfo.space/imageprocessing/histogram-equalization/
def HistEq(img):
    
    hist,bins = np.histogram(img.ravel(),256,[0,255])
    #hist = 出現次數。出現次數/總像素點 = 機率 (pdf)
    pdf = hist/img.size
    # 將每一個灰度級的機率利用cumsum()累加，變成累積機率 (cdf)。
    cdf = pdf.cumsum()
    #將cdf的結果，乘以255 (255 = 灰度範圍的最大值) ，再四捨五入，得出「均衡化值(新的灰度級)」。
    equ_value = np.around(cdf * 255).astype('uint8')
    result = equ_value[img]
    return result


fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    if i < 5:
        ax_i.imshow(HistEq(img_array_list[i]), cmap='gray')
        
        
# enhencement histogram plot
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    if i < 5:
        d = HistEq(img_array_list[i]).reshape(-1).astype(np.int)
        n, bins, patches = ax_i.hist(x=d, bins='auto', color='#0504aa',
                                     alpha=0.7, rwidth=0.85)
        maxfreq = n.max()
        ax_i.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        ax_i.grid(axis='y', alpha=0.75)
        ax_i.set_xlabel('Value')
        ax_i.set_ylabel('Frequency')
        # ax_i.set_title(r'$\mu = '+ str(round(np.mean(d),2)) + '$')
        # ax_i.imshow(img_array_list[i], cmap='gray')
fig.show()



def Local_HistEq(img, win_size):
    nrow, ncol = img.shape
    
    
    
    if win_size % 2 == 1:
        radius = int((win_size - 1) / 2)
    else:
        radius = int(win_size / 2)
        
    result = np.zeros((nrow + 2*radius, ncol + 2*radius))
    result[radius:(nrow+radius), radius:(ncol+radius)] = img
        
    for i in range(radius, nrow+radius):
        for j in range(radius, ncol+radius):
            row_index = np.arange(i-radius, i+radius+1)
            col_index = np.arange(j-radius, j+radius+1)
            
            start_row_index, end_row_index = row_index[0], row_index[-1]
            start_col_index, end_col_index = col_index[0], col_index[-1]
            
            Eq_result = HistEq(img[start_row_index:end_row_index, start_col_index:end_col_index]).reshape(-1)
            
            result[i,j] = Eq_result[int((Eq_result.shape[0] - 1) / 2)]
    return result[radius:(nrow+radius),radius:(ncol+radius)]

# https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

temp_result = hist_match(img_array_list[0], img_array_list[1])
plt.imshow(temp_result, cmap='gray')



class convolution_filter(object):
    
    def __init__(self, ):
        pass
    
    # https://datascience.stackexchange.com/questions/16625/can-the-output-of-convolution-on-image-be-higher-than-255
    def re_scale(self, image):
        min_value = image.min()
        max_value = image.max()
        normalize_image = (image - min_value) / (max_value - min_value)
        return normalize_image * 255
    
    # https://github.com/praveenVnktsh/Non-Local-Means/blob/main/main.py
    def padding(self, image, win_size):    
        height, width = image.shape
        radius = win_size // 2
        paddedImage = np.zeros((height + 2*radius, width + 2*radius)).astype(np.uint8)

        # store original image to center location
        paddedImage[radius:height + radius,radius:width + radius] = image
        
        # The next few lines creates a padded image that reflects the border
        paddedImage[radius:height+radius, 0:radius] = np.fliplr(image[:,0:radius])
        paddedImage[radius:height+radius, width+radius:width+2*radius] = np.fliplr(image[:,width-radius:width])

        paddedImage[0:radius,:] = np.flipud(paddedImage[radius:2*radius,:])
        paddedImage[height+radius:height+2*radius,:] = np.flipud(paddedImage[height:height+radius,:])
        return paddedImage
    
    # https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/gaussian_filter.py
    def gen_gaussian_kernel(self, win_size, sigma):
        center = win_size // 2
        x, y = np.mgrid[0 - center : win_size - center, 0 - center : win_size - center]
        sigma_square = np.square(sigma)
        g1 = 1 / (2 * np.pi * sigma_square)
        g2 = np.exp(-(np.square(x) + np.square(y))) / (2 * sigma_square)
        g = g1 * g2
        g = g / g.sum()
        return g
    
    def gen_averaging_kernel(self, win_size):
        g = np.ones((win_size, win_size))
        g = g / g.sum()
        return g
    
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    def gen_laplacian_of_gaussian_kernel(self, win_size, sigma):
        center = win_size // 2
        x, y = np.mgrid[0 - center : win_size - center, 0 - center : win_size - center]
        sigma_square = np.square(sigma)        
        g1 = -1 / (np.pi * np.square(sigma_square))
        g2 = 1 - (np.square(x) + np.square(y)) / (2 * sigma_square)
        g3 = np.exp(g2 - 1)        
        g = g1 * g2 * g3
        g = g / g.sum()
        return g
    
    def blur(self, image, win_size, mode, **kwargs):
        
        image = self.padding(image, win_size)
        height, width = image.shape
        # height, width = image.shape[0], image.shape[1]
        # dst image height and width
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
    
        # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
        image_array = np.zeros((dst_height * dst_width, win_size * win_size))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = np.ravel(image[i : i + win_size, j : j + win_size])
            image_array[row, :] = window
            row += 1
    
        #  turn the kernel into shape(k*k, 1)
        if mode.lower() == 'gaussian':
            kernel = self.gen_gaussian_kernel(win_size, kwargs['sigma'])
        elif mode.lower() == 'average':
            kernel = self.gen_averaging_kernel(win_size)
        elif mode.lower() == 'laplacian':
            kernel = self.gen_laplacian_of_gaussian_kernel(win_size, kwargs['sigma'])
            
        filter_array = np.ravel(kernel)
    
        # reshape and get the dst image
        un_reshape_result = np.dot(image_array.astype(np.int), filter_array.astype(np.float))
        
        result = un_reshape_result.reshape(dst_height, dst_width)
        result = self.re_scale(result).astype(np.uint8)
        return result
    
    def unsharp(self, image, win_size, mode, amount, **kwargs):
        int_img = image.astype(np.int)
        if mode.lower() == 'gaussian':
            float_blur = self.blur(image, win_size, mode, sigma=kwargs['sigma']).astype(np.float)
        elif mode.lower() == 'laplacian':
            float_blur = self.blur(image, win_size, mode, sigma=kwargs['sigma']).astype(np.float)
            
        result = int_img + (int_img - float_blur) * amount
        result = self.re_scale(result).astype(np.uint8)
        return result
    
    def denoising(self, image, win_size, mode):
        image = self.padding(image, win_size)
        height, width = image.shape
        # height, width = image.shape[0], image.shape[1]
        # dst image height and width
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
    
        # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
        image_array = np.zeros((dst_height * dst_width, win_size * win_size))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = np.ravel(image[i : i + win_size, j : j + win_size])
            image_array[row, :] = window
            row += 1
        
        if mode.lower() == 'max':
            return np.max(image_array, axis=1).reshape(dst_height, dst_width).astype(np.uint8)
        elif mode.lower() == 'median':
            return np.median(image_array, axis=1).reshape(dst_height, dst_width).astype(np.uint8)
        elif mode.lower() == 'min':
            return np.min(image_array, axis=1).reshape(dst_height, dst_width).astype(np.uint8)

    def bilateral_kernel(self, sub_image, win_size, sigma_c, sigma_s):
        center = win_size // 2
        x, y = np.mgrid[0 - center : win_size - center, 0 - center : win_size - center]
        sigma_c_square = np.square(sigma_c)
        sigma_s_square = np.square(sigma_s)
        
        gc1 = -1 / (np.pi * np.square(sigma_c_square))
        gc2 = 1 - (np.square(x) + np.square(y)) / (2 * sigma_c_square)
        gc3 = np.exp(gc2 - 1)        
        gc = gc1 * gc2 * gc3
        
        gs = np.exp(-np.square(sub_image - sub_image[center, center]) / (2 * sigma_s_square))
        
        g = gc * gs
        g = g / g.sum()
        
        return g
    
    def bilateral_filter(self, image, win_size, sigma_c, sigma_s):
        image = self.padding(image, win_size)
        height, width = image.shape
        
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
        
        result = np.zeros((dst_height, dst_width))
        for i, j in product(range(dst_height), range(dst_width)):
            sub_image = image[i : i + win_size, j : j + win_size].astype(np.int)
            kernel = self.bilateral_kernel(sub_image, win_size, sigma_c, sigma_s)
            result[i,j] = (sub_image * kernel).sum()
            
        result = self.re_scale(result).astype(np.uint8)
        return result
    
    def special_filter(self, image, mode=1):
        
        if mode == 1:
            kernel = np.array(
                [
                    [-1, 0, -1],
                    [ 0, 6,  0],
                    [-1, 0, -1],
                ])
        else:
            kernel = np.array(
                [
                    [1, 2, 1],
                    [0, 5, 0],
                    [4, 2, 4],
                ])
        win_size = 3
        
        image = self.padding(image, win_size)
        height, width = image.shape
        # dst image height and width
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
    
        # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
        image_array = np.zeros((dst_height * dst_width, win_size * win_size))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = np.ravel(image[i : i + win_size, j : j + win_size])
            image_array[row, :] = window
            row += 1
        
        filter_array = np.ravel(kernel)
    
        # reshape and get the dst image
        un_reshape_result = np.dot(image_array.astype(np.int), filter_array.astype(np.float))
        
        result = un_reshape_result.reshape(dst_height, dst_width)
        result = self.re_scale(result).astype(np.uint8)
        return result

    # https://github.com/praveenVnktsh/Non-Local-Means/blob/main/main.py

    def nonLocalMeans_filter(self, image, similarity_win_size=7, search_win_size=21, h=1, verbose=True):

        height, width = image.shape

        similarity_radius = similarity_win_size // 2
        search_radius = search_win_size // 2

        paddedImage = self.padding(image, search_win_size).astype(np.uint8)
        outputImage = paddedImage.copy()

        iterator = 0
        totalIterations = height * width * (search_win_size - similarity_win_size)**2

        if verbose:
            print("TOTAL ITERATIONS = ", totalIterations)

        for i in range(search_radius, height + search_radius):
            for j in range(search_radius, width + search_radius):
                index_i = i - search_radius
                index_j = j - search_radius

                #comparison neighbourhood
                compNbhd = paddedImage[i-similarity_radius : i+similarity_radius+1, j-similarity_radius : j+similarity_radius+1]

                pixelColor = 0
                totalWeight = 0

                for inner_i in range(index_i, index_i + search_win_size - similarity_win_size):
                    for inner_j in range(index_j, index_j + search_win_size - similarity_win_size):
                        #find the small box       
                        smallNbhd = paddedImage[inner_i:inner_i+similarity_win_size+1, inner_j:inner_j+similarity_win_size+1]
                        euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
                        #weight is computed as a weighted softmax over the euclidean distances
                        weight = np.exp(-euclideanDistance/h)
                        totalWeight += weight
                        pixelColor += weight*paddedImage[inner_i + similarity_radius, inner_j + similarity_radius]
                        iterator += 1

                        if verbose:
                            percentComplete = iterator*100/totalIterations
                            if percentComplete % 5 == 0:
                            print('% COMPLETE = ', percentComplete)

                pixelColor /= totalWeight
                outputImage[i, j] = pixelColor
        outputImage = outputImage[similarity_radius:similarity_radius+height, similarity_radius:similarity_radius+width]
        return outputImage


my_filter = convolution_filter()
my_filter.blur(img_array_list[0], 3, 'gaussian', sigma=2)
my_filter.unsharp(img_array_list[0], 3, 'gaussian', 2, sigma=2)

temp_result = my_filter.bilateral_filter(img_array_list[0], 3, 1, 2)
plt.imshow(my_filter.re_scale(image=temp_result.astype(np.int)), cmap='gray')



temp_result = my_filter.special_filter(img_array_list[0], mode=1)
plt.imshow(temp_result, cmap='gray')



fig, ax = plt.subplots(ncols=2)
ax[0].imshow(my_filter.special_filter(img_array_list[0], mode=1), cmap='gray')
ax[1].imshow(my_filter.special_filter(img_array_list[0], mode=0), cmap='gray')

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(my_filter.re_scale(image=temp_result.astype(np.int)), cmap='gray')
ax[1].imshow(img_array_list[0], cmap='gray')


