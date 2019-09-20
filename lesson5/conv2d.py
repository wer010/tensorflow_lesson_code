import tensorflow as tf
import numpy as np

kernel = tf.constant(1.0, shape=[6,6,3,1])

x3 = tf.constant(1.0, shape=[1,10,10,3])

y = tf.nn.conv2d(x3, kernel, strides=[1,1,1,1], padding="SAME")
# x3t = tf.nn.conv2d_transpose(y,kernel,output_shape=[1,5,5,3], strides=[1,2,2,1],padding="SAME")

with tf.Session() as sess:
    result = sess.run(y)
    npr = np.asarray(result)
    print(npr.shape)
    print(npr)
    # print(sess.run(x3t))