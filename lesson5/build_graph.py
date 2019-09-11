import tensorflow as tf


def build(x, y):
    w = tf.Variable(tf.zeros([2], dtype=tf.float32), name="rose_price")
    b = tf.Variable(tf.ones([1]), name="package_price")
    y_c = tf.add(tf.add(tf.multiply(x[:, 0], w[0]), tf.multiply(x[:, 1], w[1])), b, name='y_c')
    l = tf.reduce_mean(tf.square(y - y_c) / 2, name='loss')
    return y_c, l
