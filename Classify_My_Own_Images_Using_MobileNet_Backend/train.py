#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-16
#------------------------------------------------------------------------------------------#

"""
    Classify our own images using MobileNet_v2. 
    In fact this code is to show how to use pretrained popular architectures as feature 
    extractor and place multiple layers on top of them for own application. In this code we 
    used MobileNet_v2 as backend, obviously it can be replaced with others like (ResNet, ...).
    
    * The whole pipline of the project
    >>> creating train and test splits of our dataset.
        here we create an synthatic image dataset of seven symbols. Symbol names are, "spadesuit",
        "clubsuit", "Join", "Omega", "Phi", "Psi", "heartsuit". this dataset includes images 
        with diffrent sizes. obviously this dataset can be replaced with other datasets if the 
        dataset structure remain inact. Dataset structure is as follows.

        ---- dataset
                '
                '---- train
                '       '
                '       '---- category_1
                '       '       '
                '       '       '---- image_1.png
                '       '       '---- image_2.png
                '       '       '---- ....
                '       '       '---- image_n.png
                '       '
                '       '---- category_2
                '       '       '
                '       '       '---- image_1.png
                '       '       '---- image_2.png
                '       '       '---- ....
                '       '       '---- image_n.png
                '       '
                '       '----  ...
                '       '       
                '       '       
                '       '
                '       '---- category_m
                '               '
                '               '---- image_1.png
                '               '---- image_2.png
                '               '---- ....
                '               '---- image_n.png
                '       
                ' --- test 
                        '
                        '---- category_1
                        '       '
                        '       '---- image_1.png
                        '       '---- image_2.png
                        '       '---- ....
                        '       '---- image_n.png
                        '
                        '---- category_2
                        '       '
                        '       '---- image_1.png
                        '       '---- image_2.png
                        '       '---- ....
                        '       '---- image_n.png
                        '
                        '----  ...
                        '       
                        '       
                        '
                        '---- category_m
                                '
                                '---- image_1.png
                                '---- image_2.png
                                '---- ....
                                '---- image_n.png

    >>> Making a list of images and labels of train and test split.
        by running 

    >>> Loading pretrained architecture weights and placeing some layers on top of it. then train
        the model in two diffrent scenario. Finetuning backend weight or use the weight as is. these
        two scenarios will lead two different results. it's clear that tuning backend will result 
        better performance.


    In this code we used tf.data api to read images and labels from folders. And we used Keras an 
    high level api form tensorflow 2.0 to build our sequentioal model. 
    
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
                                               weights="imagenet")

base_model.trainable = cfg.FINETUNE
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(cfg.N_CLASSES, activation='softmax')

model = tf.keras.Sequential()
model.add(base_model)
model.add(global_average_layer)
model.add(prediction_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
# TODO
