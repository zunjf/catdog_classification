import tensorflow as tf 
from tensorflow import keras 
import os, cv2
from tqdm import tqdm
import numpy as np

folder_data = 'dataset/kaggle/test'

# def data_preparation(data_path):
#     test_images = []

#     for i in tqdm(os.listdir(data_path)):
#         path = os.path.join(data_path, i)
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (64, 64))
#         test_images.append([np.array(img)])
    
#     return test_images

# test_dataset = data_preparation(folder_data)
model = keras.models.load_model('save/cat_dog.h5')
model.summary()

# print(len(test_dataset))

img_path = os.path.join(folder_data, 'cat.912.jpg')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img = np.array(img).reshape(1, 64, 64, 1)

print(img.shape)
result = model.predict(img, batch_size=None, verbose=1, steps=None)

print(type(result))
print(result)
print(len(result))

if(result[0][0] > result[0][1]):
    print('cat')
else:
    print('dog')