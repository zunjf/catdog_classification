import cv2 
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf 
import matplotlib.pyplot as plt

folder_data = "dataset"
dataset_name = "kaggle"

path_data = os.path.join(folder_data, dataset_name)
path_train = os.path.join(path_data, 'train')
path_test = os.path.join(path_data, 'test')

print(path_data)
print(path_train, path_test)

def one_hot_label(img):
    label = img.split('.')[0]

    if label == 'cat':
        l = np.array([1,0])
    elif label == 'dog':
        l = np.array([0,1])
    
    return l

def train_data_with_label():
    train_images = []
    
    