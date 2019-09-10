import tensorflow as tf
import os
import build_graph

# saver = tf.train.import_meta_graph('./models/lr.ckpt-10000.meta')
# x = tf.get_default_graph().get_tensor_by_name('X1:0')
# y_c = tf.get_default_graph().get_tensor_by_name('y_c:0')



x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="X1")
y = tf.placeholder(tf.float32, name="Y")
y_c, l = build_graph.build(x, y)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './models/lr.ckpt-10000')
    re = sess.run(y_c, feed_dict={x:[[10,10]]})
    print(re)