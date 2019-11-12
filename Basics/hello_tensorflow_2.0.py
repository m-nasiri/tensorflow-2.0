#-----------------------------------------------------------------------------#
#coder: Majid Nasiri
#github: https://github.com/m-nasiri/tensorflow-2.0/1-Basics/hello_tensorflow_2.0.py
#date: 2019-November-01
#-----------------------------------------------------------------------------#

""" Hello world for tensorflow 2.0 """

import tensorflow as tf
print("tensoflow version is: %s" % tf.__version__)

x = tf.constant([[5, 2], [1, 3]])
# print(x)
# print(x.numpy())

ones = tf.ones(shape=(2, 4))
# print(ones)
zeros = tf.zeros(shape=(1,8))
# print(zeros)

rnd = tf.random.normal(shape=(1,2,4,4), mean=0.0, stddev=1.0)
# print(rnd)

var = tf.Variable(rnd)
# print(var)
# print('dtype:', var.dtype)
# print('shape:', var.shape)
# print('shape:', var.numpy)

# var.assign_add(tf.ones(shape=var.shape))
# print(var)

# var.assign(rnd+1)
# print(var)


# gradient
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))
with tf.GradientTape() as tape:
    tape.watch(a)
    c = tf.sqrt(tf.square(a) +  tf.square(b))
    dc_da = tape.gradient(c, a)
    # print(dc_da)




