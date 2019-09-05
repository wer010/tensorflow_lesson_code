import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

batch_size = 1000
epoch = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data('./mnist.npz')

X_train = np.reshape(X_train, [-1, 784])
X_test = np.reshape(X_test, [-1, 784])

data={}
data['train/image'] = X_train
data['train/label'] = y_train
data['test/image'] = X_test
data['test/label'] = y_test


def fc_layer(input, num_units,name):
    with tf.variable_scope(name):
        in_dim = input.get_shape()[1]
        W = tf.get_variable(shape=[in_dim, num_units], name='weight', dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable(shape=[num_units], name='bias', initializer=tf.zeros_initializer())
        tf.summary.histogram('weight', W)
        layer = tf.add(tf.matmul(input, W),b)
    return layer


dimensionality_train = data['train/image'].shape

num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
tf.summary.image('Image', tf.reshape(image_place,[-1,28,28,1]), max_outputs=4)

label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
label_one_hot = tf.one_hot(label_place, depth=10, axis=-1)




# global step
global_step = tf.Variable(0, name="global_step", trainable=False)

# learning rate policy
decay_steps = int(num_train_samples / batch_size * 1)
learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(0.001,
#                                            global_step,
#                                            decay_steps,
#                                            0.95,
#                                            staircase=True,
#                                            name='exponential_decay_learning_rate')
tf.summary.scalar('learning_rate', learning_rate)


z1 = fc_layer(image_place, 128, 'fc_layer_1')
o1 = tf.nn.sigmoid(z1)

z2 = fc_layer(o1, 64, 'fc_layer_2')
o2 = tf.nn.sigmoid(z2)

z3 = fc_layer(o2, 64, 'fc_layer_3')
o3 = tf.nn.sigmoid(z3)

sl = fc_layer(o3, 10,'softmax_layer')


# o1 = tf.layers.dense(inputs=image_place, units=128, name='fc_layer_1', activation=tf.nn.sigmoid)
#
# o2 = tf.layers.dense(inputs=o1, units=64, name='fc_layer_2', activation=tf.nn.sigmoid)
#
# o3 = tf.layers.dense(inputs=o2, units=10, name='output_layer')

with tf.name_scope('cost_function'):
    loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sl, labels=label_one_hot))
    tf.summary.scalar('loss', loss_tensor)

with tf.name_scope('train_op'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients_and_variables = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)
    tf.summary.histogram('gradients', gradients_and_variables[0])

with tf.name_scope('evaluation'):
    prediction_correct = tf.equal(tf.argmax(tf.nn.softmax(sl), 1), tf.argmax(label_one_hot, 1))
    # Accuracy calculation
    accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./logs_2')
    writer.add_graph(tf.get_default_graph())

    # Initialize all variables
    sess.run(tf.global_variables_initializer())


    for i in range(epoch):
        total_batch_training = int(data['train/image'].shape[0] / batch_size)

        for batch_num in range(total_batch_training):
            start_idx = batch_num * batch_size
            end_idx = (batch_num + 1) * batch_size
            # Fit training using batch data
            train_batch_data, train_batch_label = data['train/image'][start_idx:end_idx], data['train/label'][
                                                                                         start_idx:end_idx]
            summary, batch_loss, _, training_step = sess.run([merged, loss_tensor, train_op, global_step],
                feed_dict={image_place: train_batch_data, label_place: train_batch_label})
            writer.add_summary(summary, global_step=training_step)
        print("Epoch " + str(i + 1) + ", Training Loss= " + "{:.5f}".format(batch_loss))


    writer.close()


    test_accuracy = 100 * sess.run(accuracy, feed_dict={
        image_place: data['test/image'],
        label_place: data['test/label']})

    print("Final Test Accuracy is %.2f%%" % test_accuracy)