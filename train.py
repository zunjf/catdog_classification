import cv2 
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf 
import matplotlib.pyplot as plt

from tensorflow import keras

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

def data_preparation_with_label(data_path):
    train_images = []

    for i in tqdm(os.listdir(data_path)):
        path = os.path.join(data_path, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)

    return train_images

training_images = data_preparation_with_label(path_train)
testing_images  = data_preparation_with_label(path_test)

# Data
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
ts_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)

# Label
tr_label = np.array([i[1] for i in training_images])
ts_label = np.array([i[1] for i in testing_images])

# Model architecture
model = keras.Sequential()

# model.add(keras.layers.InputLayer(input_shape=[64, 64, 1]))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), 
                               padding='same', activation='relu',
                               input_shape=(64, 64, 1)))
model.add(keras.layers.MaxPool2D(pool_size=(5,5), padding='same'))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(5,5), padding='same'))

model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(5,5), padding='same'))

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# cp_path = 'save/model.ckpt'

# cp_callback = keras.callbacks.ModelCheckpoint(cp_path,
#                                               save_weights_only=True,
#                                               verbose=1)

model.fit(x=tr_img_data, y=tr_label, 
          epochs=2, validation_data=(ts_img_data, ts_label),
          batch_size=200)

loss, acc = model.evaluate(ts_img_data, ts_label)
print("trained model directly, accuracy : {:5.2f}%".format(acc*100))

model.save('save/cat_dog.h5')

converter = tf.contrib.lite.TocoConverter.from_keras_model_file("save/cat_dog.h5")
tflite_model = converter.convert()
open("save/converted_model.tflite", "wb").write(tflite_model)

# input_names = [node.op.name for node in model.inputs]
# output_names = [node.op.name for node in model.outputs]
# print(input_names, output_names)

# sess = tf.keras.backend.get_session()
# frozen_def = tf.graph_util.convert_variables_to_constants(
#     sess, sess.graph_def, output_names)

# tflite_model = tf.contrib.lite.toco_convert(
#     frozen_def, input_names, output_names)
# with tf.gfile.GFile('save/model.tflite', 'wb') as f:
#     f.write(tflite_model)
