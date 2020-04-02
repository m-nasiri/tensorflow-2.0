#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2019-November-12
#------------------------------------------------------------------------------------------#

"""
    this code is related to training a Convolutional Neural Network to clasify
    MNIST handwritten digit numbers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 
from tensorflow.keras import layers 
import numpy as np 

# Training parameters
N_ITERS = 1000     # Number of iterations
EVAL_FREQ = 10     # Evaluation after EVAL_FREQ number of iterations
N_CLASSES = 10

# Prepare a dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(32)

# Instantiate a simple classification model
class CNN_Model(tf.keras.Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2_1 = layers.Dense(N_CLASSES, activation='softmax')
        self.fc2_2 = layers.Dense(N_CLASSES, activation='softmax')
        self.fc2_3 = layers.Dense(N_CLASSES, activation='softmax')

    def call(self, inputs):
        net = self.conv1(inputs)
        net = self.flatten(net)
        net = self.fc1(net)
        out_1 = self.fc2_1(net)
        out_2 = self.fc2_2(net)
        out_3 = self.fc2_3(net)
        return out_1, out_2, out_3

model = CNN_Model()

# define loss object
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# define mean metric for taining loss
train_loss = tf.keras.metrics.Mean(name="train_loss") 
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

# define mean metric for testing loss
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


# first custom loss
def custom_dist_loss(y_true, y_pred):
    y_true = tf.one_hot(y_true, depth=N_CLASSES)
    loss_value = tf.math.reduce_mean(tf.math.abs(y_true - y_pred))
    return loss_value

# second custom loss
def custom_mse_loss(y_true, y_pred):
    y_true = tf.one_hot(y_true, depth=N_CLASSES)
    loss_value = tf.keras.losses.MSE(y_true, y_pred)
    return loss_value

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        y_pred_1, y_pred_2, y_pred_3 = model(images)
        xent_loss_value = loss_obj(labels, y_pred_1)
        custom_dist_loss_value = custom_dist_loss(labels, y_pred_2)
        custom_mse_loss_value = custom_mse_loss(labels, y_pred_3)

        loss = xent_loss_value + custom_dist_loss_value + custom_mse_loss_value
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, y_pred_1)


@tf.function
def test_step(images, labels):
    y_pred_1, _, _ = model(images)
    t_loss = loss_obj(labels, y_pred_1)
    test_loss(t_loss)
    test_accuracy(labels, y_pred_1)


# Train for N_ITERS
for itr in range(N_ITERS):
    train_images, train_labels = next(iter(train_ds))
    train_step(train_images, train_labels)

    # Evaluating
    if ((itr+1)% EVAL_FREQ) == 0:
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print_str = 'Iter {}, Train loss: {:.2f}, Train accuracy: {:.2f}, Test loss: {:.2f}, Test accuracy: {:.2f}'
        print(print_str.format(itr+1,
                               train_loss.result(),
                               train_accuracy.result(),
                               test_loss.result(),
                               test_accuracy.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

