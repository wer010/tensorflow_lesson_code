import tensorflow as tf

def build_graph(X_1, X_2,Y_):
    w1 = tf.Variable(0.0, name="red_rose_price")
    w2 = tf.Variable(0.0, name="white_rose_price")
    b = tf.Variable(1.0, name="package_price")

    Y_c = w1 * X_1 + w2 * X_2 + b

    l = tf.reduce_mean(tf.square(Y_ - Y_c) / 2)
    return Y_c,l
