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
model.compile(optimizer=optimizer, loss=loss)

model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.3)
# save model and weights
model.save(filepath="model.h5")
y_pred = model.predict(x=x_test)
y_pred = np.argmax(y_pred, axis=1)
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred, num_classes=10)
print(cm)


# load model
model2 = tf.keras.models.load_model("model.h5")
# print(model2.summary())
y_pred = model2.predict(x=x_test)
y_pred = np.argmax(y_pred, axis=1)
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred, num_classes=10)
print(cm)
