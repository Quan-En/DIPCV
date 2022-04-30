

import os
from tqdm import tqdm

import numpy as np
from numpy import array, zeros
import pandas as pd

import cv2

from BATCH_DCT_DETECT import DCT_DETECT

my_dct_detect = DCT_DETECT()

train_data_path = "/home/re6091054/FF++/c40/train"
image_types = os.listdir(train_data_path) # 5 type images


path_dict = dict()

for img_type in image_types:
    full_type_name = train_data_path + "/" + img_type
    folder_name_list = sorted(os.listdir(full_type_name))
    path_dict[img_type] = folder_name_list



category_dict = {"Real":0, "NeuralTextures":1, "Face2Face":2, "Deepfakes":3, "FaceSwap":4}


def GetFeature(path_list):
    result_x = []
    result_y = []
    for path in tqdm(path_list):
        temp_image_list = []
        for image_name in os.listdir(path):
            if "png" in image_name:
                temp_image_list.append(cv2.imread(path + "/" + image_name))
        result_x.append(my_dct_detect.main(temp_image_list))
        
        img_type = path.split("/")[-2]
        result_y.append(array([category_dict[img_type]] * len(temp_image_list)))
        
    return result_x, result_y

def GetPathList(index):
    path_list = [train_data_path + "/" + img_type + "/" + path_dict[img_type][index] for img_type in image_types]
    return path_list


def GetPlotDF(arr_list):
    result_df = pd.DataFrame(np.column_stack([np.concatenate(arr_list[1]), np.row_stack(arr_list[0])]))
    result_df = result_df.sort_values(0, ascending=False).reset_index(drop=True)
    result_df[0] = result_df[0].astype(int)
    return result_df


def GetBalanceTrainArr(arr_list):
    x_arr_list, y_arr_list = arr_list
    real_image_num = len(y_arr_list[2])
    each_type_num = real_image_num // 4
    for i in [0,1,3,4]:
        x_arr_list[i] = x_arr_list[i][:each_type_num,:]
        y_arr_list[i] = y_arr_list[i][:each_type_num]
    
    x_arr = np.row_stack(x_arr_list)
    y_arr = np.concatenate(y_arr_list)
    y_arr[y_arr != 0] = 1
    return x_arr, y_arr