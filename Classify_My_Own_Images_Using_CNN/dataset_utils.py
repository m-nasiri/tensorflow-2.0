#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-09
#------------------------------------------------------------------------------------------#

""" 
    functions to read images and labels for train and test portion of dataset. 
    These function uses tf.data api from tensorflow 2.0 in order to read, 
    prefetch and batch data for model. 
"""

import tensorflow as tf 
import os

N_CLASSES = 10

def get_filenames_list(data_dir, is_training):
    if is_training:
        fh = open(os.path.join(data_dir, "train_images_list.txt"), "r")
        image_list = [line.rstrip() for line in fh.readlines()]
        fh = open(os.path.join(data_dir, "train_labels_list.txt"), "r")
        label_list = [int(line.rstrip()) for line in fh.readlines()]
    else:
        fh = open(os.path.join(data_dir, "test_images_list.txt"), "r")
        image_list = [line.rstrip() for line in fh.readlines()]
        fh = open(os.path.join(data_dir, "test_labels_list.txt"), "r")
        label_list = [int(line.rstrip()) for line in fh.readlines()]
    return image_list, label_list

def _parse_function(image_paths, labels, is_training):
    image_string = tf.io.read_file(image_paths)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_decoded.set_shape([28, 28, 3])
    image_decoded = image_decoded / 255
    labels = tf.one_hot(labels, depth=N_CLASSES)
    return image_decoded, labels

def input_fn(data_dir, is_training, batch_size):
    image_list, label_list = get_filenames_list(data_dir=data_dir, is_training=is_training)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    if is_training:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(lambda image,label: _parse_function(image, label, is_training))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(None) # repeat forever
    dataset = dataset.batch(batch_size)

    return dataset

