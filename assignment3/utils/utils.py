import numpy as np
from numpy import array


def re_scale(image):
    min_value, max_value = image.min(), image.max()
    result = 255 * (image - min_value) / (max_value - min_value)
    return result.astype(np.uint8)

def freq_padding(image:np.array, new_size=None):
    nrows, ncols = image.shape[0], image.shape[1]

    if new_size is None:
        new_size = np.power(2, np.floor(np.log2([nrows, ncols])+1)).astype(int)
        
    output = np.zeros(new_size)
    output[:nrows,:ncols] = image
    return output

# https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy
# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
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


def img_shift(x):
    nrows, ncols = x.shape[0], x.shape[1]
    row_center_index = int(nrows / 2)
    col_center_index = int(ncols / 2)

    block_array_list = [
        x[:row_center_index,:col_center_index],
        x[:row_center_index,col_center_index:],
        x[row_center_index:,:col_center_index],
        x[row_center_index:,col_center_index:],
        ]
    # block_array_list = list(map(lambda x: np.flip(x, (0,1)), block_array_list))
    # shifted_result = np.block([block_array_list[:2], block_array_list[2:]])
    shifted_result = np.block([
        [block_array_list[3], block_array_list[2]],
        [block_array_list[1], block_array_list[0]],
    ])
    return shifted_result


def create_image_radius_index(size:tuple):
    nrows, ncols = size[0], size[1]
    all_row_index, all_col_index = np.indices((nrows, ncols))
    row_center_index = int(nrows / 2)
    col_center_index = int(ncols / 2)
    all_row_index = all_row_index - row_center_index
    all_col_index = all_col_index - col_center_index
    return all_row_index, all_col_index