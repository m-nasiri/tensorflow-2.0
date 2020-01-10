#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-09
#------------------------------------------------------------------------------------------#

"""
    In this code we use tf.data api to read images and labels from folders. 
    At the first step we extract MNIST dataset images and store all images in train and test 
    folders and we prepare a list of train and test images and also train and test labels by
    running extract_mnist_images.py 
    After running the script we must have a folder structured like following in the main 
    folder beside the script. train_images_list.txt contain path to images in train folder and 
    test_images_list.txt contain path to images in test folder.

    --- dataset
           |
           |-- train
           |    |__ image_0.png
           |    |__ image_1.png
           |    |__ ...
           |    |__ image_N.png
           |
           |-- test
           |    |__ image_0.png
           |    |__ image_1.png
           |    |__ ...
           |    |__ image_N.png
           |
           |-- test_images_list.txt
           |-- test_labels_list.txt
           |-- train_images_list.txt
           |-- train_labels_list.txt

    You can prepare your own dataset as our structure. 
    dataset_utils.py contain functions to read images and labels for train and test portion of
    dataset. These function uses tf.data api to read, prefetch and batch data for model.

    mnist_tfdata_cnn.py is the main code to read images and labels, building a CNN model and feed 
    train and test data to model for training and evaluation. 

    We used Keras an high level api form tensorflow 2.0 to build our sequentioal model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 
from tensorflow.keras import layers
import dataset_utils
import numpy as np 
import matplotlib.pyplot as plt


DATA_DIR = "./dataset"
BATCH_SIZE = 250
N_EPOCHS = 1
n_train = 60000
n_test = 10000
steps_per_epoch = n_train // BATCH_SIZE

# train dataset queue using tf.data
train_datset = dataset_utils.input_fn(data_dir=DATA_DIR, is_training=True, batch_size=BATCH_SIZE)

# test dataset queue using tf.data
test_dataset = dataset_utils.input_fn(data_dir=DATA_DIR, is_training=False, batch_size=BATCH_SIZE)

# define a convolutional model with 3 convolution and 2 dense layers.
model = tf.keras.Sequential([
                    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28 ,3)),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Flatten(),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(10, activation='softmax')
                   ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x=train_datset, epochs=N_EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=test_dataset, validation_steps=100)

# model.evaluate(test_dataset, steps=1000)
# model.save(filepath="model.h5")

# try an image
import cv2
im = cv2.imread("./dataset/test/image_0.png") / 255.0
img = np.expand_dims(im, axis=0)
pred = model.predict_classes(img)

plt.imshow(im)
plt.title("prediction: %d" % pred)
plt.show()
