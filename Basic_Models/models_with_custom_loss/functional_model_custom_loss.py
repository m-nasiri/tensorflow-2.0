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


def FUNCTIONAL_MODEL():
    inputs = tf.keras.Input(shape=(None, None, 1), name='input')
    net = layers.Conv2D(4,(3,3))(inputs)
    net1 = layers.MaxPool2D((2,2), (2,2))(net)
    net2 = layers.Conv2D(4,(3,3), padding="SAME")(net1)
    net = tf.concat([net1, net2], axis=-1)
    net = layers.Conv2D(120,(3,3))(net)
    net = layers.GlobalAveragePooling2D()(net)
    outputs = layers.Dense(10, activation="softmax")(net)
    model = tf.keras.Model(inputs, outputs, name='func_api')
    return model

# define custom loss
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.uint8)
    y_true = tf.squeeze(y_true)
    y_true_oh = tf.one_hot(y_true, depth=10, axis=-1)
    xent = - y_true_oh * tf.math.log(y_pred)
    xent = tf.math.reduce_mean(xent)
    return xent

# Instantiate the model.
model = FUNCTIONAL_MODEL()

# loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(loss=custom_loss, optimizer=optimizer, metrics=metrics)
model.fit(x=x_train, y=y_train, validation_split=0.3)
# save functional model
model.save("functional_model.h5")


# load functional model
model_loaded = tf.keras.models.load_model("functional_model.h5", compile=False)
print(model_loaded.summary())



