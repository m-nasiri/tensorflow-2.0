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

# Instantiate the model.
model = FUNCTIONAL_MODEL()


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
