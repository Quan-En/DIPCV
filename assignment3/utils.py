import numpy as np
from numpy import array


def re_scale(image):
    min_value, max_value = image.min(), image.max()
    result = 255 * (image - min_value) / (max_value - min_value)
    return result.astype(np.uint8)

def freq_padding(image:np.array):
    nrows, ncols = image.shape[0], image.shape[1]
    new_size = np.power(2, np.floor(np.log2([nrows, ncols])+1)).astype(int)
    output = np.zeros(new_size)
    output[:nrows,:ncols] = image
    return output

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))