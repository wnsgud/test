import os, os.path
import cv2
import torch

image_sort = ['png','jpg','jpeg']
image_list = []


path = './temp_data'
for filename in os.listdir(path):

    for i in image_sort:
        if filename.lower().endswith(i):

            file_path = path + '/' + filename
            image = cv2.imread(file_path)
            image_list.append(image)

print(image_list)
