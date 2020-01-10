#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-09
#------------------------------------------------------------------------------------------#

"""
    In this script we extract MNIST dataset images and store all images in train and test 
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

"""

import os
import numpy as np
import tensorflow as tf 
import cv2

if not os.path.exists("./dataset/train"): 
    os.makedirs("./dataset/train")

if not os.path.exists("./dataset/test"): 
    os.makedirs("./dataset/test")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
n_train = x_train.shape[0]
n_test = x_test.shape[0]

print("Writing train images ...")
images = np.reshape(x_train, [-1, 28, 28, 1])
fh_image = open("./dataset/train_images_list.txt", "w")
fh_label = open("./dataset/train_labels_list.txt", "w")
for i in range(n_train):
    image_name = "image_%d.png" % i
    cv2.imwrite("./dataset/train/" + image_name, images[i])
    fh_image.write("./dataset/train/" + image_name)
    fh_image.write("\n")
    fh_label.write(str(y_train[i]))
    fh_label.write("\n")
fh_image.close()
fh_label.close()


print("Writing test images ...")
images = np.reshape(x_test, [-1, 28, 28, 1])
fh_image = open("./dataset/test_images_list.txt", "w")
fh_label = open("./dataset/test_labels_list.txt", "w")
for i in range(n_test):
    image_name = "image_%d.png" % i
    cv2.imwrite("./dataset/test/" + image_name, images[i])
    fh_image.write("./dataset/test/" + image_name)
    fh_image.write("\n")
    fh_label.write(str(y_test[i]))
    fh_label.write("\n")
fh_image.close()
fh_label.close()

print("Done!")


