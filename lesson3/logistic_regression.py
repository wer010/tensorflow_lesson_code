import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

batch_size = 500
epoch = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data('./mnist.npz')

X_train = np.reshape(X_train, [-1, 784])
X_test = np.reshape(X_test, [-1, 784])

data={}
data['train/image'] = X_train
data['train/label'] = y_train
data['test/image'] = X_test
data['test/label'] = y_test

def extract_samples_Fn(data):
    index_list = []
    for sample_index in range(data.shape[0]):
        label = data[sample_index]
        if label == 1 or label == 0:
            index_list.append(sample_index)
    return index_list


index_list_train = extract_samples_Fn(data['train/label'])
index_list_test = extract_samples_Fn(data['test/label'])

data['train/image'] = X_train[index_list_train]
data['train/label'] = y_train[index_list_train]

data['test/image'] = X_test[index_list_test]
data['test/label'] = y_test[index_list_test]

dimensionality_train = data['train/image'].shape

num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
tf.summary.image('Image', tf.reshape(image_place,[-1,28,28,1]), max_outputs=4)

label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
label_one_hot = tf.one_hot(label_place, depth=2 , axis=-1)




# global step
global_step = tf.Variable(0, name="global_step", trainable=False)

# learning rate policy
decay_steps = int(num_train_samples / batch_size * 1)
learning_rate = tf.train.exponential_decay(0.001,
                                           global_step,
                                           decay_steps,
                                           0.95,
                                           staircase=True,
                                           name='exponential_decay_learning_rate')
tf.summary.scalar('learning_rate', learning_rate)


with tf.name_scope('variable'):
    w = tf.Variable(tf.zeros([num_features, 2]), name='weight')
    b = tf.Variable(tf.zeros([2]), name='bias')
    tf.summary.histogram('weight', w)

with tf.name_scope('cost_function'):
    logits = tf.matmul(image_place, w) + b
    loss_tensor = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
    tf.summary.scalar('loss', loss_tensor)

with tf.name_scope('train_op'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients_and_variables = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)
    tf.summary.histogram('gradients', gradients_and_variables[0])

with tf.name_scope('evaluation'):
    prediction_correct = tf.equal(tf.argmax(tf.nn.sigmoid(logits), 1), tf.argmax(label_one_hot, 1))
    # Accuracy calculation
    accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:

    saver = tf.train.Saver()
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