#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2019-November-12
#------------------------------------------------------------------------------------------#

"""
    this code is related to reloading HDF5 file and reusing the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 
from tensorflow.keras import layers 
import numpy as np 

# Prepare a dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model.h5')

# Show the model architecture
model.summary()

loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

