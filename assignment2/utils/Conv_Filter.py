# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:45:47 2021

@author: Taner
"""


import cv2
import numpy as np
from itertools import product
from tqdm import tqdm

class Conv_Filter(object):
    
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
        g2 = np.exp(-(np.square(x) + np.square(y)) / (2 * sigma_square))
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

    def gen_laplacian_kernel(self, win_size):
        def ff(n1, n2):
            c = 1
            for i in range(n1, n2+1):
                c *= i
            return c
        s = win_size - 1
        temp = []
        for i in range(0, win_size):
            c = (ff(i+1, s) / ff(1, s-i)) * ((-1) ** i)
            temp.append(c)
        temp = np.array(temp)
        p1 = np.zeros([win_size, win_size])
        p1[int(win_size / 2), :] = temp
        f = p1 + p1.T
        return f

    def laplacian_filter(self, image, win_size):
        image = self.padding(image, win_size).astype(np.float64)
        height, width = image.shape

        radius = win_size // 2
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
        
        result = np.zeros((dst_height, dst_width))

        kernel = self.gen_laplacian_kernel(win_size)

        for i in range(radius, radius+dst_height):
            for j in range(radius, radius+dst_width):
                result[i-radius, j-radius] = (image[i-radius:i+radius+1, j-radius:j+radius+1] * kernel).sum()

        result = self.re_scale(result).astype(np.uint8)
        return result
    
    def blur(self, image, win_size, mode, **kwargs):
        
        image = self.padding(image, win_size).astype(np.float64)
        height, width = image.shape
        # height, width = image.shape[0], image.shape[1]
        # dst image height and width
        radius = win_size // 2
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
        
        result = np.zeros((dst_height, dst_width))

        if mode.lower() == 'gaussian':
            kernel = self.gen_gaussian_kernel(win_size, kwargs['sigma'])
        elif mode.lower() == 'average':
            kernel = self.gen_averaging_kernel(win_size)
        elif mode.lower() == 'laplacian':
            kernel = self.gen_laplacian_of_gaussian_kernel(win_size, kwargs['sigma'])

        for i in range(radius, radius+dst_height):
            for j in range(radius, radius+dst_width):
                result[i-radius, j-radius] = (image[i-radius:i+radius+1, j-radius:j+radius+1] * kernel).sum()

        result = self.re_scale(result).astype(np.uint8)
        return result
    
    def unsharp(self, image, win_size, mode, amount, **kwargs):
        int_img = image.astype(np.float64)
        if mode.lower() == 'gaussian':
            float_blur = self.blur(image, win_size, mode, sigma=kwargs['sigma']).astype(np.float64)
        elif mode.lower() == 'laplacian':
            float_blur = self.blur(image, win_size, mode, sigma=kwargs['sigma']).astype(np.float64)
            
        result = int_img + (int_img - float_blur) * amount
        result = self.re_scale(result).astype(np.uint8)
        return result
    
    def gen_sobel_kernel(self, win_size, dir=1):   # 1為計算橫向，2為直向
        s = int(win_size / 2) + 1
        l = np.linspace(1, win_size, win_size)
        x, y = np.meshgrid(l, l)
        xy = np.concatenate([np.expand_dims(x, 0), np.expand_dims(y, 0)], axis=0)
        if dir == 1:
            dirs = np.array([1, 0])
        else:
            dirs = np.array([0, 1])

        center = np.array([int(win_size/2), int(win_size/2)]).reshape(2, 1, 1) + 1
        temp = (xy - center)
        distence = np.abs(temp).sum(axis=0).reshape(1, win_size, win_size)
        f = temp * dirs.reshape(2, 1, 1) / distence
        f = f.sum(axis=0)
        f[s-1, s-1] = np.array([0])
        return f * (win_size - 1)
    
    def sobel_filter(self, image, win_size, dir=1):
        image = self.padding(image, win_size).astype(np.float64)
        height, width = image.shape
        # height, width = image.shape[0], image.shape[1]
        # dst image height and width
        radius = win_size // 2
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
        
        result = np.zeros((dst_height, dst_width))

        kernel = self.gen_sobel_kernel(win_size, dir)
        for i in range(radius, radius+dst_height):
            for j in range(radius, radius+dst_width):
                result[i-radius, j-radius] = (image[i-radius:i+radius+1, j-radius:j+radius+1] * kernel).sum()

        result = self.re_scale(result).astype(np.uint8)
        return result

    def denoising(self, image, win_size, mode):
        radius = win_size // 2
        image = self.padding(image, win_size)
        height, width = image.shape
        # height, width = image.shape[0], image.shape[1]
        # dst image height and width
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1

        result = np.zeros((dst_height, dst_width))

        for i in range(radius, radius+dst_height):
            for j in range(radius, radius+dst_width):
                if mode.lower() == 'max':
                    result[i-radius, j-radius] = np.max(image[i-radius:i+radius+1, j-radius:j+radius+1])
                elif mode.lower() == 'median':
                    result[i-radius, j-radius] = np.median(image[i-radius:i+radius+1, j-radius:j+radius+1])
                else:
                    result[i-radius, j-radius] = np.min(image[i-radius:i+radius+1, j-radius:j+radius+1])

        return result.astype(np.uint8)
        # # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
        # image_array = np.zeros((dst_height * dst_width, win_size * win_size))
        # row = 0
        # for i, j in product(range(dst_height), range(dst_width)):
        #     window = np.ravel(image[i : i + win_size, j : j + win_size])
        #     image_array[row, :] = window
        #     row += 1
        
        # if mode.lower() == 'max':
        #     return np.max(image_array, axis=1).reshape(dst_height, dst_width).astype(np.uint8)
        # elif mode.lower() == 'median':
        #     return np.median(image_array, axis=1).reshape(dst_height, dst_width).astype(np.uint8)
        # elif mode.lower() == 'min':
        #     return np.min(image_array, axis=1).reshape(dst_height, dst_width).astype(np.uint8)

    def gen_bilateral_kernel(self, sub_image, win_size, sigma_c, sigma_s, without_smooth=False):
        center = win_size // 2
        x, y = np.mgrid[0 - center : win_size - center, 0 - center : win_size - center]
        sigma_c_square = np.square(sigma_c)
        sigma_s_square = np.square(sigma_s)

        if not without_smooth:
            gc1 = -1 / (np.pi * np.square(sigma_c_square))
            gc2 = 1 - (np.square(x) + np.square(y)) / (2 * sigma_c_square)
            gc3 = np.exp(gc2 - 1)        
            gc = gc1 * gc2 * gc3
        else:
            gc = 1
        
        gs = np.exp(-np.square(sub_image - sub_image[center, center]) / (2 * sigma_s_square))
        
        g = gc * gs
        g = g / g.sum()
        
        return g
    
    def bilateral_filter(self, image, win_size, sigma_c, sigma_s, without_smooth=False):
        image = self.padding(image, win_size)
        height, width = image.shape
        
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
        
        result = np.zeros((dst_height, dst_width))
        for i, j in product(range(dst_height), range(dst_width)):
            sub_image = image[i : i + win_size, j : j + win_size].astype(np.float64)
            kernel = self.gen_bilateral_kernel(sub_image, win_size, sigma_c, sigma_s, without_smooth)
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
        radius = win_size // 2
        image = self.padding(image, win_size)
        height, width = image.shape
        # dst image height and width
        dst_height = height - win_size + 1
        dst_width = width - win_size + 1
        result = np.zeros((dst_height, dst_width))


        for i in range(radius, radius+dst_height):
            for j in range(radius, radius+dst_width):
                result[i-radius, j-radius] = (image[i-radius:i+radius+1, j-radius:j+radius+1] * kernel).sum()

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

        for i in tqdm(range(search_radius, height + search_radius)):
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
                        smallNbhd = paddedImage[inner_i:inner_i+similarity_win_size, inner_j:inner_j+similarity_win_size]
                        euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
                        #weight is computed as a weighted softmax over the euclidean distances
                        weight = np.exp(-euclideanDistance/h)
                        totalWeight += weight
                        pixelColor += weight*paddedImage[inner_i + similarity_radius, inner_j + similarity_radius]
                        iterator += 1

                        if verbose:
                            percentComplete = iterator*100/totalIterations
                            # if percentComplete // 5 == 0:
                                # print('% COMPLETE = ', percentComplete)
                            # if percentComplete % 5 == 0:
                                # print('% COMPLETE = ', percentComplete)

                pixelColor /= totalWeight
                outputImage[i, j] = pixelColor
        outputImage = outputImage[similarity_radius:similarity_radius+height, similarity_radius:similarity_radius+width]
        return outputImage
