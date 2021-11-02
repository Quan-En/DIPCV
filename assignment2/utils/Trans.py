import numpy as np


class Transform(object):
    def __init__(self, ):
        pass
    
    def re_scale(self, image):
        min_value = image.min()
        max_value = image.max()
        normalize_image = (image - min_value) / (max_value - min_value)
        return normalize_image * 255
    
    def log_trans(self, image):
        result = 46 * np.log(np.array(image, dtype=np.int32) + 1)
        return np.round(result).astype(np.uint8)
    
    def gamma_trans(self, image, power=1):
        return np.array(255 * ((np.array(image, dtype=np.int32)/255) ** power), dtype=np.uint8)
    
    def neg_trans(self, image):
        return np.array(255 - np.array(image, dtype=np.int32), dtype=np.uint8)
    
    def image_resize(self, image, height, width, mode=1):
        if mode == 1:
            return self.bilinear_resize_vectorized(image, height, width)
        else:
            return self.nearest_neighbor_resize_vectorized(image, height, width)
    
    def bilinear_resize_vectorized(self, image, height, width):
        """
        `image` is a 2-D numpy array
        `height` and `width` are the desired spatial dimension of the new 2-D array.
        """
        image = image.astype(np.int32)
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
      
        return traget.reshape(height, width).astype(np.uint8)
      
    def nearest_neighbor_resize_vectorized(self, image, height, width):
        """
        `image` is a 2-D numpy array
        `height` and `width` are the desired spatial dimension of the new 2-D array.
        """
        image = image.astype(np.int32)
        img_height, img_width = image.shape
      
        image = image.ravel()
      
        x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
        y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
      
        y, x = np.divmod(np.arange(height * width), width)
      
        x_c = np.round(x_ratio * x).astype('int32')
        y_c = np.round(y_ratio * y).astype('int32')
        traget = image[y_c * img_width + x_c]
      
        return traget.reshape(height, width).astype(np.uint8)