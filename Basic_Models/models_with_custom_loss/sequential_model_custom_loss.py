#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2019-March-28
#------------------------------------------------------------------------------------------#

import tensorflow as tf 
import tensorflow.keras.layers as layers 
import numpy as np 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255

# Sequential API
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(4, (3,3), padding="SAME", input_shape=(28,28,1)))
model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.Dropout(0.2)) # drop rate
model.add(layers.Dense(10, activation="softmax"))

# define custom loss
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.uint8)
    y_true = tf.squeeze(y_true)
    y_true_oh = tf.one_hot(y_true, depth=10, axis=-1)
    xent = - y_true_oh * tf.math.log(y_pred)
    xent = tf.math.reduce_mean(xent)
    return xent

# print(model.weights)
# print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_loss)
model.fit(x=x_train, y=y_train, epochs=3, validation_split=0.3)
