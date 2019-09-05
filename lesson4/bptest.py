import tensorflow as tf
import numpy as np
a = tf.placeholder(tf.float32, shape=([1,2]))
b = tf.placeholder(tf.float32, shape=([1,1]))
w1= tf.Variable([[0.1,0.2],[0.3,0.4]],name='w1',dtype=tf.float32)
b1=tf.Variable([-0.3,0.3],name='b1',dtype=tf.float32)
z1 = tf.nn.sigmoid(tf.add(tf.matmul(a,w1),b1))
w2= tf.Variable([[0.6],[0.7]],name='w2',dtype=tf.float32)
b2=tf.Variable([[0.5]],name='b2',dtype=tf.float32)
z2 = tf.nn.sigmoid(tf.add(tf.matmul(z1,w2),b2))

loss = tf.square(b-z2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
gradients_and_variables = optimizer.compute_gradients(loss)

train_op = optimizer.apply_gradients(gradients_and_variables)

x = np.array([[0.5,0.5]])
y = np.array([[0.0]])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
l1,l2,g_v = sess.run([z1,z2,gradients_and_variables],feed_dict={a: x, b: y})

print(l1)
print(l2)
print(g_v)

sess.run([train_op],feed_dict={a: x, b: y})

print(sess.run(w1))
print(sess.run(w2))
print(sess.run(b1))
print(sess.run(b2))
print(sess.run(z2,feed_dict={a: x, b: y}))
# print(sess.run([z1,z2,gradients_and_variables],feed_dict={a: x, b: y}))

