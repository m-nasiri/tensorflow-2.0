#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2019-November-12
#------------------------------------------------------------------------------------------#

"""
    this code is related to training a Convolutional Neural Network to clasify MNIST 
    handwritten digit numbers. we save entire model for further usage. 
    Saving a fully-functional model is very useful—you can load them in TensorFlow.js 
    (HDF5, Saved Model) and then train and run them in web browsers, or convert them 
    to run on mobile devices using TensorFlow Lite (HDF5, Saved Model)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 
from tensorflow.keras import layers 
import numpy as np 

# Training parameters
N_ITERS = 100     # Number of iterations
EVAL_FREQ = 10     # Evaluation after EVAL_FREQ number of iterations

# Prepare a dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(32)


# Instantiate a simple classification model
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# define loss object
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# define mean metric for taining loss
train_loss = tf.keras.metrics.Mean(name="train_loss") 
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

# define mean metric for testing loss
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_obj(labels, predictions)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_obj(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


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

# save the a model's architecture, weights, and training configuration in a single file/folder.
# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model shuold be saved to HDF5.
model.save('model.h5')



