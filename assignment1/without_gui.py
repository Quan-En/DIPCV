# Packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image file
bmp_img_path_list = ["baboon.bmp", "boat.bmp", "F16.bmp"]
raw_img_path_list = ["goldhill.raw", "lena.raw", "peppers.raw"]


img_array_list = []
for f_name in bmp_img_path_list:
    img_array_list.append(cv2.imread(f_name, cv2.IMREAD_GRAYSCALE))
for f_name in raw_img_path_list:
    img_array_list.append(np.fromfile(f_name, dtype=np.uint8).reshape(512, 512))


# Enhencement function

def log_trans(img):
    return np.array(np.log(np.array(img, dtype=np.int32) + 1), dtype=np.uint8)

def gamma_trans(img,power=0.1):
    return np.array(255 * ((np.array(img, dtype=np.int32)/255) ** power), dtype=np.uint8)

def neg_trans(img):
    return np.array(255 - np.array(img, dtype=np.int32), dtype=np.uint8)

# Resize function
## Bilinear resize:
## Reference: https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
def bilinear_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_lower = np.floor(x_ratio * x).astype('int32')
  y_lower = np.floor(y_ratio * y).astype('int32')

  x_upper = np.ceil(x_ratio * x).astype('int32')
  y_upper = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_lower
  y_weight = (y_ratio * y) - y_lower

  traget_a = image[y_lower * img_width + x_lower]
  traget_b = image[y_lower * img_width + x_upper]
  traget_c = image[y_upper * img_width + x_lower]
  traget_d = image[y_upper * img_width + x_upper]

  traget = traget_a * (1 - x_weight) * (1 - y_weight) + \
           traget_b * x_weight * (1 - y_weight) + \
           traget_c * y_weight * (1 - x_weight) + \
           traget_d * x_weight * y_weight

  return traget.reshape(height, width)

## Nearest neighbor resize:
## Reference: https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
def nearest_neighbor_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_c = np.round(x_ratio * x).astype('int32')
  y_c = np.round(y_ratio * y).astype('int32')
  traget = image[y_c * img_width + x_c]

  return traget.reshape(height, width)


# Plot
    
## plot 1: original
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(img_array_list[i], cmap='gray')

## plot 2: center 10 by 10
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(img_array_list[i][251:261, 251:261], cmap='gray')
    
## plot 3: log transform
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(log_trans(img_array_list[i]), cmap='gray')

## plot 4: gamma transform
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(gamma_trans(img_array_list[i], 1.5), cmap='gray')

## plot 5: negative transform
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(neg_trans(img_array_list[i]), cmap='gray')

## plot 6: bilinear resize (512, 512) -> (128, 128)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(bilinear_resize_vectorized(img_array_list[i], 128, 128), cmap='gray')

## plot 7: bilinear resize (512, 512) -> (32, 32)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(bilinear_resize_vectorized(img_array_list[i], 32, 32), cmap='gray')

## plot 8: bilinear resize (512, 512) -> (32, 32) -> (512, 512)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(bilinear_resize_vectorized(bilinear_resize_vectorized(img_array_list[i], 32, 32), 512, 512), cmap='gray')

## plot 9: bilinear resize (512, 512) -> (1024, 512)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(bilinear_resize_vectorized(img_array_list[i], 1024, 512), cmap='gray')
    
## plot 10: bilinear resize (512, 512) -> (128, 128) -> (256, 512)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(bilinear_resize_vectorized(bilinear_resize_vectorized(img_array_list[i], 128, 128), 256, 512), cmap='gray')

## plot 11: nearest neighbor resize (512, 512) -> (128, 128)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(nearest_neighbor_resize_vectorized(img_array_list[i], 128, 128), cmap='gray')

## plot 12: nearest neighbor resize (512, 512) -> (32, 32)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(nearest_neighbor_resize_vectorized(img_array_list[i], 32, 32), cmap='gray')

## plot 13: nearest neighbor resize (512, 512) -> (32, 32) -> (512, 512)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(nearest_neighbor_resize_vectorized(nearest_neighbor_resize_vectorized(img_array_list[i], 32, 32), 512, 512), cmap='gray')

## plot 14: nearest neighbor resize (512, 512) -> (1024, 512)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(nearest_neighbor_resize_vectorized(img_array_list[i], 1024, 512), cmap='gray')

## plot 15: nearest neighbor resize (512, 512) -> (128, 128) -> (256, 512)
fig, ax = plt.subplots(nrows=2, ncols=3)
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(nearest_neighbor_resize_vectorized(nearest_neighbor_resize_vectorized(img_array_list[i], 128, 128), 256, 512), cmap='gray')
