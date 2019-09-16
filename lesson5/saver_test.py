import tensorflow as tf
import numpy as np
import os
from lesson5 import build_graph

n = 1000
X1 = np.random.randint(0, 2 * n, size=n)
X2 = np.random.randint(0, 2 * n, size=n)
data = np.stack((X1, X2), 1)
price = 10 * X1 + 8 * X2 + 15
num_epochs = 10000

x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="X1")
y = tf.placeholder(tf.float32, name="Y")
y_c, l = build_graph.build(x, y)

learning_rate = 0.0005
t = tf.train.AdamOptimizer(learning_rate).minimize(l, name='train_op')
saver = tf.train.Saver()
save_path = 'models/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with tf.Session() as sess:
    # 初始化变量

    sess.run(tf.global_variables_initializer())
    w = tf.get_default_graph().get_tensor_by_name('rose_price:0')
    b = tf.get_default_graph().get_tensor_by_name('package_price:0')
    # 训练模型
    for epoch_num in range(num_epochs):
        loss_value, _ = sess.run([l, t], feed_dict={x: data, y: price})
        # 每训练5000步显示一下当前的loss
        if (epoch_num+1) % 5000 is 0:
            print('epoch %d, loss=%f' % (epoch_num+1, loss_value))
            s_p = saver.save(sess, './models/lr.ckpt', global_step=epoch_num+1)
            print(s_p)
            print(sess.run([w, b]))
            print(sess.run(y_c, feed_dict={x: [[10, 10]]}))
