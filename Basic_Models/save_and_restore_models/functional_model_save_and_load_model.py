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


class FUNCTIONAL_MODEL(tf.keras.Model):
    def __init__(self):
        super(FUNCTIONAL_MODEL, self).__init__()
        self.conv1 = layers.Conv2D(4,(3,3))
        self.maxpool1 = layers.MaxPool2D((2,2), (2,2))
        self.conv2 = layers.Conv2D(8,(3,3))
        self.maxpool2 = layers.MaxPool2D((2,2), (2,2))
        self.conv3 = layers.Conv2D(120,(3,3))
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(10, activation="softmax")

    def call(self, x):
        net = self.conv1(x)
        net1 = self.maxpool1(net)
        net = self.conv2(net1)
        net = self.maxpool2(net)
        net = self.conv3(net)
        net = self.avgpool(net)
        net = self.fc(net)
        return net

    def get_model(self):
        inputs = tf.keras.Input(shape=(None, None, 1), name='input')
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs, outputs, name='func_api')
        return model



# Instantiate the model.
model = FUNCTIONAL_MODEL().get_model()


loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(x=x_train, y=y_train, validation_split=0.3)
# save functional model
model.save("functional_model.h5")


# load functional model
model_loaded = tf.keras.models.load_model("functional_model.h5")
print(model_loaded.summary())



