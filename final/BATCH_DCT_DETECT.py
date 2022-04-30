import numpy as np
from scipy.fftpack import dct, idct
import cv2

class DCT_DETECT(object):
    def __init__(self, name='DCT_detect'):
        self.name = name
        self.ZigZag_index()

    def main(self, img_list):
        img_list = list(map(lambda x: x.astype(float), img_list))
        img = self.Padd2SameSize(img_list)
        rgb_img = self.BGR2RGB(img)
        padd_img = self.ZeroPad(rgb_img)
        blocks_img = self.Split2Blocks(padd_img)
        dct_blocks = self.DctBlocks(blocks_img)
        blocks_zigzag_result = self.toZigZag(dct_blocks)
        
#         outputs = self.BetaEstimate(blocks_zigzag_result)
#         return outputs
        
        padd_all_zeros_indices = np.all(blocks_zigzag_result == 0, axis=2)
        
        beta_result_list = []
        for sub_zigzag_result, sub_all_zeros_index in zip(blocks_zigzag_result, padd_all_zeros_indices):
            beta = np.abs(sub_zigzag_result[~sub_all_zeros_index,:]).mean(axis=0)
            beta_result_list.append(beta[1:])
            
        return np.stack(beta_result_list)
    
    def Padd2SameSize(self, img_list):
        all_size = np.row_stack(list(map(lambda x: x.shape, img_list)))
        max_size = all_size.max(axis=0)
        N = all_size.shape[0]
        result = np.zeros((N, *max_size))
        for i in range(N):
            height, width, channels = img_list[i].shape
            result[i][:height,:width,:] = img_list[i]
        return result
    
    def BGR2RGB(self, img):
        return img[:, :, :, ::-1]
    
    def ZeroPad(self, img):  # N * H * W * C
        height = img.shape[1]
        width = img.shape[2]
        # padding image to (8 \times c1, 8 \times c2)
        new_height = 8 - (height % 8)
        new_width = 8 - (width % 8)
        result = np.pad(img, [(0, 0), (0, new_height), (0, new_width), (0, 0)], mode='constant', constant_values=0)
        return result

    def Split2Blocks(self, img):
        N, H, W, C = img.shape
        h, w = H // 8, W // 8
        outputs = img.reshape(N, h, 8, w, 8, C)
        outputs = outputs.transpose((0, 2, 4, 1, 3, 5))  # N * 8 * 8 * h * w * c
        return outputs

    def DctBlocks(self, inputs):# N * 8 * 8 * h * w * c
        inputs = inputs.transpose((0, 3, 4, 5, 2, 1))
        outputs = dct(inputs, norm = 'ortho').transpose((0, 1, 2, 3, 5, 4))
        outputs = dct(outputs, norm = 'ortho')
        outputs = outputs.transpose((0, 4, 5, 1, 2, 3))
        return outputs

    def ZigZag_index(self):
        ind = np.arange(0, 64).reshape(8, 8)
        lines=[[] for i in range(8+8-1)]
        for i in range(8):
            for j in range(8):
                s = i + j
                if(s % 2 == 0):
                    lines[s].insert(0, ind[i, j])
                else:
                    lines[s].append(ind[i, j])
        ind = np.array([i for line in lines for i in line])
        self.ZigZag = ind.astype(int)

    def toZigZag(self, inputs):  # N * 8 * 8 * h * w * c
        N = inputs.shape[0]
        inputs = inputs.transpose((0, 3, 4, 5, 1, 2))  # N * h * w * c * 8 * 8
        inputs = inputs.reshape(N, -1, 64)
        outputs = inputs[:, :, self.ZigZag]
        return outputs

    def BetaEstimate(self, inputs):  # N, -1, 64
        outputs = np.abs(inputs).mean(axis=1)[:, 1:]
        return outputs # only return AC