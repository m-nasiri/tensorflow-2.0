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

#print(model.weights)
#print(model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
acc = tf.keras.metrics.Accuracy()
model.compile(optimizer=optimizer, loss=loss)

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