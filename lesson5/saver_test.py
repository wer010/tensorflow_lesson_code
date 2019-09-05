import tensorflow as tf
import numpy as np
import os

n = 1000
X1 = np.random.randint(0, 2*n, size=n)
X2 = np.random.randint(0, 2*n, size=n)
data = np.stack((X1,X2),1)
price = 10*X1+ 8*X2 +15
num_epochs=60000



X = tf.placeholder(shape=[None,2], dtype=tf.float32, name="X1")
Y_ = tf.placeholder(tf.float32, name="Y")

w = tf.Variable(tf.zeros([2],dtype=tf.float32), name="red_rose_price")
b = tf.Variable(tf.ones([1]), name="package_price")

Y_c = tf.add(tf.add(tf.multiply(X[:,0],w[0]), tf.multiply(X[:,1],w[1])),b, name='y_c')
l = tf.reduce_mean(tf.square(Y_ - Y_c) / 2)


learning_rate = 0.0005
t = tf.train.AdamOptimizer(learning_rate).minimize(l)
saver = tf.train.Saver()
save_path = 'models/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with tf.Session() as sess:
    # 初始化变量

    sess.run(tf.global_variables_initializer())

    # 训练模型
    for epoch_num in range(num_epochs):
        loss_value, _ = sess.run([l,t], feed_dict={X: data, Y_: price})
        # 每训练5000步显示一下当前的loss
        if epoch_num%5000 is 0:
            print('epoch %d, loss=%f' %(epoch_num, loss_value))
            s_p = saver.save(sess,'./models/lr.ckpt',global_step=epoch_num)
            print(s_p)
            print(sess.run([w, b]))
            print(sess.run(Y_c, feed_dict={X: [[10,10]]}))


