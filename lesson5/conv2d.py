import tensorflow as tf
import numpy as np

kernel = tf.constant(1.0, shape=[6,6,3,1])

x3 = tf.constant(1.0, shape=[1,10,10,3])
y3 = tf.constant(1.0, shape=[1,5,5,1])

y = tf.nn.conv2d(x3, kernel, strides=[1,2,2,1], padding="SAME")
x3t = tf.nn.conv2d_transpose(y3,kernel,output_shape=[1,10,10,3], strides=[1,2,2,1],padding="SAME")

def print_shape(x):
    npr = np.asarray(x)
    print(npr.shape)
    print(npr)

with tf.Session() as sess:
    x = sess.run(y)
    print_shape(x)
    x = sess.run(x3t)
    print_shape(x)

