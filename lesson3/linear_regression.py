import numpy as np
import tensorflow as tf

# 生成数据
n = 1000
X1 = np.random.randint(0, 2*n, size=n)
X2 = np.random.randint(0, 2*n, size=n)
Y= 10*X1+ 8*X2 +15

num_epochs=60000

# 创建tensorflow计算图

with tf.name_scope('Input'):
    X_1 = tf.placeholder(tf.float32, name="X1")
    X_2 = tf.placeholder(tf.float32, name="X2")
    Y_ = tf.placeholder(tf.float32, name="Y")
with tf.name_scope('Variable'):
    w1 = tf.Variable(0.0, name="red_rose_price")
    tf.summary.scalar('red_rose_price', w1)
    w2 = tf.Variable(0.0, name="white_rose_price")
    tf.summary.scalar('white_rose_price', w2)
    b = tf.Variable(1.0, name="package_price")
    tf.summary.scalar('package_price', b)

# Y_c = w1 * X_1 + w2 * X_2 + b
with tf.name_scope('Result'):
    Y_c = tf.add(tf.add(w1 * X_1 , w2 * X_2), b, name = 'Y_c')
    l = tf.reduce_mean(tf.square(Y_ - Y_c) / 2, name = 'loss')
    tf.summary.scalar('loss', l)

learning_rate = 0.0005
with tf.name_scope('train'):
    t = tf.train.AdamOptimizer(learning_rate).minimize(l)
# 在这里加入三行代码
writer = tf.summary.FileWriter('./logs')
writer.add_graph(tf.get_default_graph())
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # 初始化变量

    sess.run(tf.global_variables_initializer())

    # 训练模型
    for epoch_num in range(num_epochs):
        summary, loss_value, _ = sess.run([merged,l,t], feed_dict={X_1: X1, X_2: X2, Y_: Y})
        # 每训练5000步显示一下当前的loss
        if epoch_num%50 is 0:
            writer.add_summary(summary,epoch_num)
            print('The epoch num is %d' %epoch_num)

writer.close()
