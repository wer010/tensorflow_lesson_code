import tensorflow as tf
import numpy as np
import os
import build_graph

n = 1000
X1 = np.random.randint(0, 2 * n, size=n)
X2 = np.random.randint(0, 2 * n, size=n)
data = np.stack((X1, X2), 1)
price = 10 * X1 + 8 * X2 + 15
num_epochs = 10000


saver = tf.train.import_meta_graph('./models/lr.ckpt-10000.meta')
x = tf.get_default_graph().get_tensor_by_name('X1:0')
y_c = tf.get_default_graph().get_tensor_by_name('y_c:0')
t = tf.get
epoch_num = 10000

with tf.Session() as sess:
    saver.restore(sess, './models/lr.ckpt-10000')
    re = sess.run(y_c, feed_dict={x: [[10, 10]]})
    print(re)
    for epoch_num in range(num_epochs+1):
        loss_value, _ = sess.run([l, t], feed_dict={x: data, y: price})
        # 每训练5000步显示一下当前的loss
        if epoch_num % 5000 is 0:
            print('epoch %d, loss=%f' % (epoch_num, loss_value))
            s_p = saver.save(sess, './models/lr.ckpt', global_step=epoch_num)
            print(s_p)
            print(sess.run([w, b]))
            print(sess.run(y_c, feed_dict={x: [[10, 10]]}))