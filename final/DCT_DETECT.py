"""
DCT detection of DeepFake image

Reference: Fighting Deepfakes by Detecting GAN DCT Anomalies (2021)


"""



import numpy as np
from numpy import array, zeros

from scipy.fftpack import dct, idct



class DCT_DETECT(object):
    
    def __init__(self):
        pass

    def main(self, img):
        rgb_img = self.BGR2RGB(img)
        padd_img = self.ZeroPad(rgb_img)
        blocks_img = self.Split2Blocks(padd_img)
        dct_blocks = self.DctBlocks(blocks_img)
        blocks_zigzag_result = self.BlocksZigZag(dct_blocks)
        beta_result = self.BetaEstimate(blocks_zigzag_result)
        return beta_result

    def BGR2RGB(self, img):
        return img[:,:,::-1]
    
    def ZeroPad(self, img):
        height = img.shape[0]
        width = img.shape[1]
        # padding image to (8 \times c1, 8 \times c2)
        new_height = height + (8 - (height % 8))
        new_width = width + (8 - (width % 8))
        result = zeros((new_height, new_width, 3), dtype=np.uint8)
        result[:height,:width,:] = img
        return result

    def Split2Blocks(self, img):
        yLen = img.shape[0] // 8
        xLen = img.shape[1] // 8
        blocks = zeros((yLen, xLen, 8, 8, 3), dtype=np.uint8)
        for y in range(yLen):
            for x in range(xLen):
                blocks[y][x] = img[y*8:(y+1)*8, x*8:(x+1)*8]
        return array(blocks)


    def DctBlocks(self, blocks):
        xLen = blocks.shape[1]
        yLen = blocks.shape[0]
        result = zeros((yLen, xLen, 8, 8, 3))
        for y in range(yLen):
            for x in range(xLen):
                d = zeros((8, 8, 3))
                for i in range(3):
                    block = blocks[y][x][:,:,i]
                    d[:,:,i] = dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
                result[y][x] = d
        return result

    def ZigZag(self, block):
        lines=[[] for i in range(8+8-1)]
        for y in range(8):
            for x in range(8):
                i = y + x
                if(i % 2 == 0):
                    lines[i].insert(0, block[y][x])
                else:
                    lines[i].append(block[y][x])
        return array([coefficient for line in lines for coefficient in line])

    def BlocksZigZag(self, blocks):
        xLen = blocks.shape[1]
        yLen = blocks.shape[0]
        zz = zeros(xLen * yLen * 3, dtype=object)
        for y in range(yLen):
            for x in range(xLen):
                for i in range(3):
                    zz[y * xLen * 3 + x * 3 + i] = self.ZigZag(blocks[y,x,:,:,i])
        return zz

    def BetaEstimate(self, blocks_zigzag_result):
        return np.abs(np.stack(blocks_zigzag_result)).mean(axis=0)[1:] # only return AC