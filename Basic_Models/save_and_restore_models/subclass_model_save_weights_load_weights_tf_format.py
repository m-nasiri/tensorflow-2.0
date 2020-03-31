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
# x_train = x_train[0:100]
# y_train = y_train[0:100]

# custom layer
class Linear(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, name):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'), name=name + "kernel")
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), name=name + "bias")

    def call(self, x):
        return tf.matmul(x, self.w) + self.b

# custom layer
class ResLayer(tf.keras.layers.Layer):
    def __init__(self, filters, name):
        super(ResLayer, self).__init__()
        self.conv_a = layers.Conv2D(filters, (3,3), padding="SAME", activation="relu", name=name + "conv_a")
        self.conv_b = layers.Conv2D(filters, (3,3), padding="SAME", activation="relu", name=name + "conv_b")

    def call(self, x):
        net_a = self.conv_a(x)
        net_b = self.conv_b(net_a)
        return net_a + net_b

# subclass model
class SUBCLASS_MODEL(tf.keras.Model):
    def __init__(self):
        super(SUBCLASS_MODEL, self).__init__(name="mymodel")
        self.conv1 = layers.Conv2D(4,(3,3))
        self.maxpool1 = layers.MaxPool2D((2,2), (2,2))
        self.reslayer_1 = ResLayer(filters=12, name="mymodel/ResLayer/")
        self.maxpool2 = layers.MaxPool2D((2,2), (2,2))
        self.conv3 = layers.Conv2D(120,(3,3))
        self.avgpool = layers.GlobalAveragePooling2D()
        self.linear_1 = Linear(units=50, input_dim=120, name="mymodel/Linear/")
        self.fc = layers.Dense(10, activation="softmax")

    def call(self, x):
        net = self.conv1(x)
        net = self.maxpool1(net)
        net = self.reslayer_1(net)
        net = self.maxpool2(net)
        net = self.conv3(net)
        net = self.avgpool(net)
        net = self.linear_1(net)
        net = self.fc(net)
        return net


# Instantiate the model.
model = SUBCLASS_MODEL()

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
acc = tf.keras.metrics.Accuracy()
model.compile(loss=loss, optimizer=optimizer)


# train one epoch
model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save weights after first epoch
model.save_weights(filepath="model/subclass_model_weights_1", save_format="tf")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for first epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric


# train another epoch
model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save weights after second epoch
model.save_weights(filepath="model/subclass_model_weights_2", save_format="tf")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for second epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric


# train another epoch
model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save weights after third epoch
model.save_weights(filepath="model/subclass_model_weights_3", save_format="tf")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print(acc.result().numpy())
print("Accuracy for third epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric



# load different weights 
# load saved weights from first epoch 
model.load_weights("model/subclass_model_weights_1")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for loaded weight from first epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric

# load saved weights from second epoch 
model.load_weights("model/subclass_model_weights_2")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for loaded weight from second epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric

# load saved weights from third epoch 
model.load_weights("model/subclass_model_weights_3")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for loaded weight from third epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric


# print(model.conv1.weights)
# print(model.reslayer_1.conv_a.weights)

""" It does not work for subclassed models """
# model.save("sm.h5")
# model_loaded = tf.keras.models.load_model("sm.h5")
