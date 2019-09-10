import tensorflow as tf

def build(X ,Y_):
    w = tf.Variable(tf.zeros([2], dtype=tf.float32), name="red_rose_price")
    b = tf.Variable(tf.ones([1]), name="package_price")
    Y_c = tf.add(tf.add(tf.multiply(X[:, 0], w[0]), tf.multiply(X[:, 1], w[1])), b, name='y_c')
    l = tf.reduce_mean(tf.square(Y_ - Y_c) / 2)
    return Y_c,l
