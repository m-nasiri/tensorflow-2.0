#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2018-Feb-04
#------------------------------------------------------------------------------------------#

"""
    this code is related to training a convolutional neural network to clasify
    images contain 7 symbols (spadesuit, clubsuit, Join, Omega, Phi, Psi, 
    heartsuit)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import tensorflow as tf
import numpy as np
import os
import dataset_mean

# Constants used for dealing with the files
CLASS_NAMES = ["spadesuit", "clubsuit", "Join", "Omega", "Phi", "Psi", "heartsuit"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_WIDTH = 36
IMG_HEIGHT = 36
DATASET_MEAN = np.array([231.212, 231.633, 231.959], dtype=np.float32)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# tf.debugging.set_log_device_placement(True)
# tf.test.is_gpu_available()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./dataset')
parser.add_argument('--train_dir', default='./dataset/train')
parser.add_argument('--test_dir', default='./dataset/test')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_iters', default=10000, type=int)
parser.add_argument('--ckpt_dir', default='./ckpt/')

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def train_preprocess(file_path):
    # get label
    label = get_label(file_path)
    label = tf.dtypes.cast(label, tf.float32)
    # turn to integer label
    label = tf.argmax(label)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Take a random 36x36 crop to the scaled image
    img = tf.image.random_crop(img, [IMG_WIDTH, IMG_HEIGHT, 3])
    # Horizontally flip the image with probability 1/2
    img = tf.image.random_flip_left_right(img)
    # Substract the per color mean `DATASET_MEAN`
    means = tf.reshape(tf.constant(DATASET_MEAN), [1, 1, 3])
    img = img - means

    return img, label

def test_preprocess(file_path):
    # get label
    label = get_label(file_path)
    label = tf.dtypes.cast(label, tf.float32)
    # turn to integer label
    label = tf.argmax(label)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Take a random 36x36 crop to the scaled image
    img = tf.image.random_crop(img, [IMG_WIDTH, IMG_HEIGHT, 3])
    # Substract the per color mean `DATASET_MEAN`
    means = tf.reshape(tf.constant(DATASET_MEAN), [1, 1, 3])
    img = img - means

    return img, label

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPool2D(padding="same")
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPool2D(padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc2 = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")

    def call(self, inputs):
        net = self.conv1(inputs)
        net = self.maxpool1(net)
        net = self.conv2(net)
        net = self.maxpool2(net)
        net = self.flatten(net)
        net = self.fc1(net)
        net = self.fc2(net)
        return net


def main(args):

    # Training dataset
    train_list_ds = tf.data.Dataset.list_files(args.train_dir + "/*/*", shuffle=True) # shuffle image list
    train_labeled_ds = train_list_ds.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    train_labeled_ds.cache()
    # shuffle next 100 samples in queue
    train_labeled_ds = train_labeled_ds.shuffle(buffer_size=100) 
    train_labeled_ds = train_labeled_ds.repeat()
    train_labeled_ds = train_labeled_ds.batch(args.batch_size)
    train_labeled_ds = train_labeled_ds.prefetch(buffer_size=AUTOTUNE)

    # Test dataset
    test_list_ds = tf.data.Dataset.list_files(args.test_dir + "/*/*")
    test_labeled_ds = test_list_ds.map(test_preprocess, num_parallel_calls=AUTOTUNE)
    test_labeled_ds.cache()
    test_labeled_ds = test_labeled_ds.batch(args.batch_size)
    test_labeled_ds = test_labeled_ds.prefetch(buffer_size=AUTOTUNE)

    # Define Convolutional Network Graph
    model = CNN()

    # loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    # Use tf.GradientTape to train the model.
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    """Train CNN for a number of steps."""    
    for itr in range(args.n_iters):        
        train_images, train_labels = next(iter(train_labeled_ds))
        train_step(train_images, train_labels)

        if (itr+1)%100 == 0:
            for test_images, test_labels in test_labeled_ds:
                test_step(test_images, test_labels)

            template = 'Iter {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}'
            print(template.format(itr+1,
                                  train_loss.result(),
                                  train_accuracy.result()*100,
                                  test_loss.result(),
                                  test_accuracy.result()*100))


            test_acc = test_accuracy.result()*100
            # Reset the metrics for the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

        if ((itr+1)%1000 == 0):
            # Save the weights using the `checkpoint_path` format
            model.save_weights(os.path.join(args.ckpt_dir, "model_itr={itr:04d}_acc={acc:.2f}.ckpt".format(itr=itr+1, acc=test_acc)))




if __name__ == '__main__':
    args = parser.parse_args()

    # evaluate mean vector value for RGB images. Each mean value for each layer.
    DATASET_MEAN = dataset_mean.mean(args.dataset_dir)
    DATASET_MEAN = np.array(DATASET_MEAN, dtype=np.float32) / 255
    print("DATASET_MEAN:", DATASET_MEAN)


    main(args)
