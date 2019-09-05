import tensorflow as tf
import os


saver = tf.train.import_meta_graph('./models/lr.ckpt-55000.meta')
saver = tf.train.Saver()
x = tf.get_default_graph().get_tensor_by_name('X1:0')
y = tf.get_default_graph().get_tensor_by_name('y_c:0')

with tf.Session() as sess:
    saver.restore(sess, './models/lr.ckpt-55000')
    re = sess.run(y,feed_dict={x:[[10,10]]})
    print(re)