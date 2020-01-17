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
from config import Config as cfg

AUTOTUNE = tf.data.experimental.AUTOTUNE

steps_per_epoch = cfg.N_TRAIN // cfg.BATCH_SIZE

# train dataset queue using tf.data
train_datset = dataset_utils.input_fn(is_training=True, batch_size=cfg.BATCH_SIZE)

# test dataset queue using tf.data
test_dataset = dataset_utils.input_fn(is_training=False, batch_size=1)


# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=(cfg.IMG_HIGHT, cfg.IMG_WIDTH, 3),
                                               include_top=False,
                                               weights="mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5")

base_model.trainable = True
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(cfg.N_CLASSES, activation='softmax')

model = tf.keras.Sequential()
model.add(base_model)
model.add(global_average_layer)
model.add(prediction_layer)

model.compile(optimizer="adam", 
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=["accuracy"])

print(model.summary())

history = model.fit(train_datset,
                    epochs=cfg.N_EPOCHS, 
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=cfg.N_TEST,
                    validation_data=test_dataset)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

# # try an image
# import cv2
# im = cv2.imread("./dataset/test/image_0.png") / 255.0
# img = np.expand_dims(im, axis=0)
# pred = model.predict_classes(img)

# plt.imshow(im)
# plt.title("prediction: %d" % pred)
# plt.show()
