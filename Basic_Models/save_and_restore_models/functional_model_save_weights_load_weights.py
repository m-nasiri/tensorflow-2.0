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
        inputs = tf.keras.Input(shape=(None, None, 1), name='img')
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs, outputs, name='func_api')
        return model



# Instantiate the model.
model = FUNCTIONAL_MODEL().get_model()


loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
acc = tf.keras.metrics.Accuracy()
model.compile(loss=loss, optimizer=optimizer)

# train one epoch
model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save weights after first epoch
model.save_weights(filepath="functional_model_weights_1.h5")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for first epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric

# train another epoch
model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save weights after second epoch
model.save_weights(filepath="functional_model_weights_2.h5")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for second epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric


# train another epoch
model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save weights after third epoch
model.save_weights(filepath="functional_model_weights_3.h5")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print(acc.result().numpy())
print("Accuracy for third epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric



# load different weights 
# load saved weights from first epoch 
model.load_weights("functional_model_weights_1.h5")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for loaded weight from first epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric

# load saved weights from second epoch 
model.load_weights("functional_model_weights_2.h5")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for loaded weight from second epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric

# load saved weights from third epoch 
model.load_weights("functional_model_weights_3.h5")
y_pred = model.predict(x=x_test) # evaluate model on test split
y_pred = np.argmax(y_pred, axis=1)
acc.update_state(y_test, y_pred) # calculate accuracy metric
print("Accuracy for loaded weight from third epoch:", acc.result().numpy())
acc.reset_states() # reset accuracy metric
