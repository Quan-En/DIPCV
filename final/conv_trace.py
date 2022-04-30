
import numpy as np
from math import floor


import torch
import torch.nn as nn





class Conv_trace(object):
    
    def E_step(self, I, K, sigma):
        # I: image 2D matrix
        # K: kernel matrix
        # sigma: sigma of gaussian function
        
        R = self.R_f(I, K)
        P = self.P_f(R, sigma)

        p0 = 1 / (I.max() - I.min())
        W = self.W_f(P, p0)
        P_use = p0/(p0+P)**2
        
        return R, P_use, P, W
    
    def M_step(self, I, R, P_use, P, W, K):
        # I: image 2D matrix
        # W: probability matrix
        # K: kernel matrix

        # (m_rows, n_cols) in original image
        I_rows, I_cols = I.shape
        # radius of kernel
        radius = floor(K.shape[0] / 2)
        # padding image
        padded_I = self.padding_f(I, radius)
        # make sure center of K equal to zero
        K[radius,radius] = 0
        # new K
        new_K = K[:]
        # all [i,j] index of kernel K
        kernel_index_x, kernel_index_y = np.meshgrid(np.arange(2*radius+1), np.arange(2*radius+1))
        kernel_index = np.column_stack([kernel_index_x.reshape(-1), kernel_index_y.reshape(-1)])

        for index, (i, j) in enumerate(kernel_index):

            I_plus_i_plus_j = padded_I[i:(i+I_rows), j:(j+I_cols)]
            W_prime = W * I_plus_i_plus_j

            denominator = np.sum(W_prime * I_plus_i_plus_j)

            numerator = np.sum(W_prime * I)
            for s,t in np.delete(kernel_index, (index), axis=0):
                I_plus_s_plus_t = padded_I[s:(s+I_rows), t:(t+I_cols)]
                numerator -= K[s,t] * np.sum(W_prime * I_plus_s_plus_t)
                
            new_K[i,j] = numerator / denominator
            
        # make sure center of new K equal to zero
        new_K[radius,radius] = 0
        
        #sigma
        sigma2 = sum(R**2*P_use*P*R**2)/sum(P_use*P*R**2)
        sigma = np.sqrt(sigma2)

        return new_K, sigma
    
    def padding_f(self, I, radius):
        I_rows, I_cols = I.shape
        padded_I =  np.zeros((I_rows + 2 * radius, I_cols + 2 * radius))
        padded_I[radius:(I_rows+radius),radius:(I_cols+radius)] = I
        return padded_I
    
    def torch_Conv_f(self, I, K):
        I_rows, I_cols = I.shape
        radius = K.shape[0] // 2
        result = np.zeros((I_rows, I_cols))
        padded_I = self.padding_f(I, radius)
        
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=radius, stride=1, padding=radius, bias=False)
        pass
    
    
    def Conv_f(self, I, K):
        I_rows, I_cols = I.shape
        radius = K.shape[0] // 2
        result = np.zeros((I_rows, I_cols))
        padded_I = self.padding_f(I, radius)

        for i in range(radius, radius+I_rows):
            for j in range(radius, radius+I_cols):
                result[i-radius, j-radius] = (padded_I[i-radius:i+radius+1, j-radius:j+radius+1] * K).sum()
        return result

    def R_f(self, I, K):
        return np.abs(I - self.Conv_f(I, K))

    def P_f(self, R, sigma):
        left_side = 1 / (sigma * np.sqrt(2 * np.pi))
        right_side = np.exp(-R**2 / (2 * sigma**2))
        # right_side = np.exp(-(R**2) / (2 * sigma**2))
        return left_side * right_side

    def W_f(self, P, p0):
        W = P / (P + p0)
        return W
    
    def K(self, sub_im):
        #sub_im = sub_im[:,:,i].astype(float)
        sub_im = sub_im.astype(float)
        K = np.random.rand(3,3)
        K[1, 1] = 0
        sigma = 1
        i = 0
        while(True):
            i+=1
            print(i, end=',')
            R, P_use, P, W = self.E_step(sub_im, K, sigma)
            new_K, new_sigma = self.M_step(sub_im, R, P_use, P, W, K)
            if np.sum(new_K-K)<0.001 or i>100:
                break
            else:
                K = new_K
                sigma = new_sigma
        return new_K
    
    
    def get_K(self, img):
        features = []
        for i in range(3):
            sub_im = img[:,:,i].astype(float)
            new_K = self.K(sub_im)
            K_feature = np.delete(new_K, 4)
            features.extend(K_feature)
        return features