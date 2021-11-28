# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:43:38 2021

@author: Taner
"""

import cv2
import numpy as np

class Hist_Enhencement(object):
    def __init__(self, ):
        pass

    # https://datascience.stackexchange.com/questions/16625/can-the-output-of-convolution-on-image-be-higher-than-255
    def re_scale(self, image):
        min_value = image.min()
        max_value = image.max()
        normalize_image = (image - min_value) / (max_value - min_value)
        return normalize_image * 255

    # https://codeinfo.space/imageprocessing/histogram-equalization/
    def HistEq(self, img):
        
        hist,bins = np.histogram(img.ravel(),256,[0,255])
        #hist = 出現次數。出現次數/總像素點 = 機率 (pdf)
        pdf = hist/img.size
        # 將每一個灰度級的機率利用cumsum()累加，變成累積機率 (cdf)。
        cdf = pdf.cumsum()
        #將cdf的結果，乘以255 (255 = 灰度範圍的最大值) ，再四捨五入，得出「均衡化值(新的灰度級)」。
        equ_value = np.around(cdf * 255).astype('uint8')
        result = equ_value[img]
        return result
    
    def Local_HistEq(self, img, win_size):
        img = img.astype(np.float64)
        nrow, ncol = img.shape
        radius = win_size // 2

        g_mean = np.mean(img)
        g_var = np.var(img)
        k_0 = 0.7
        k_1 = 0.02
        k_2 = 0.7
        E = 4

        result = np.zeros((nrow + 2*radius, ncol + 2*radius))
        result[radius:(nrow+radius), radius:(ncol+radius)] = img
        
        # The next few lines creates a padded image that reflects the border
        result[radius:nrow+radius, 0:radius] = np.fliplr(img[:,0:radius])
        result[radius:nrow+radius, ncol+radius:ncol+2*radius] = np.fliplr(img[:,ncol-radius:ncol])

        result[0:radius,:] = np.flipud(result[radius:2*radius,:])
        result[nrow+radius:nrow+2*radius,:] = np.flipud(result[nrow:nrow+radius,:])

        for i in range(radius, nrow+radius):
            for j in range(radius, ncol+radius):
                kernel = result[i-radius:i+radius+1, j-radius:j+radius+1]
                l_mean = np.mean(kernel)
                l_var = np.var(kernel)
                if l_mean <= k_0*g_mean and k_1*g_var < l_var < k_2*g_var:
                    result[i,j] = E*result[i,j]
                
        result = self.re_scale(result[radius:(nrow+radius),radius:(ncol+radius)]).astype(np.uint8)
        return result


    # def Local_HistEq(self, img, win_size):
    #     nrow, ncol = img.shape
    #     radius = win_size // 2
            
    #     result = np.zeros((nrow + 2*radius, ncol + 2*radius))
    #     result[radius:(nrow+radius), radius:(ncol+radius)] = img
        
    #     # The next few lines creates a padded image that reflects the border
    #     result[radius:nrow+radius, 0:radius] = np.fliplr(img[:,0:radius])
    #     result[radius:nrow+radius, ncol+radius:ncol+2*radius] = np.fliplr(img[:,ncol-radius:ncol])

    #     result[0:radius,:] = np.flipud(result[radius:2*radius,:])
    #     result[nrow+radius:nrow+2*radius,:] = np.flipud(result[nrow:nrow+radius,:])
            
    #     for i in range(radius, nrow+radius):
    #         for j in range(radius, ncol+radius):
                
    #             Eq_result = self.HistEq(img[i-radius:i+radius+1, j-radius:j+radius+1]).reshape(-1)
                
    #             result[i,j] = Eq_result[Eq_result.shape[0] // 2]
    #     return result[radius:(nrow+radius),radius:(ncol+radius)]
    
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    def hist_match(self, source, template):
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


if __name__ == '__main__':
    pass