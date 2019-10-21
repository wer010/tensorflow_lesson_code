import tensorflow as tf
import numpy as np

n = 1000
X1 = np.random.randint(0, 2 * n, size=n)
X2 = np.random.randint(0, 2 * n, size=n)
data = np.stack((X1, X2), 1)
price = 10 * X1 + 8 * X2 + 15
num_epochs = 5000

saver_name = tf.train.latest_checkpoint('./models/')

saver = tf.train.import_meta_graph(saver_name+'.meta')
g = tf.get_default_graph()
x = g.get_tensor_by_name('X1:0')
y = g.get_tensor_by_name('Y:0')
y_c = g.get_tensor_by_name('y_c:0')
l = g.get_tensor_by_name('loss:0')
w = g.get_tensor_by_name('rose_price:0')
b = g.get_tensor_by_name('package_price:0')
t = g.get_operation_by_name('train_op')

with tf.Session() as sess:
    start_step = int(saver_name.split('-')[-1])

    saver.restore(sess, saver_name)
    re = sess.run(y_c, feed_dict={x: [[10, 10]]})
    print(re)
    for epoch_num in range(start_step, start_step+num_epochs):
        loss_value, _ = sess.run([l, t], feed_dict={x: data, y: price})
        # 每训练5000步显示一下当前的loss
        if (epoch_num+1) % 5000 is 0:
            print('epoch %d, loss=%f' % (epoch_num+1, loss_value))
            s_p = saver.save(sess, './models/lr.ckpt', global_step=epoch_num+1)
            print(s_p)
            print(sess.run([w, b]))
            print(sess.run(y_c, feed_dict={x: [[10, 10]]}))