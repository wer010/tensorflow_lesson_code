import tensorflow as tf
import wave
import numpy as np
import cv2
def main():


    a= tf.Variable(1, name='a')
    tf.summary.scalar('a',a)

    b= tf.Variable(tf.random_normal([100]),name='b')
    # tf.summary.scalar('b',b)
    tf.summary.histogram('b',b)

    c = tf.constant(np.zeros([100]), name='c')
    tf.summary.histogram('c',c)

    with wave.open('./Ring01.wav', 'rb') as f:
        params = f.getparams()
        str_data = f.readframes(params[3])
        wave_data = np.fromstring(str_data, dtype=np.short)
        wave_data = wave_data.reshape([1 , -1 , 2])/65536.0
        wave_data = np.tile(wave_data,(5,1,1))
        tf.summary.audio('d', wave_data, sample_rate=22050)

    img = cv2.imread('test.jpg')[np.newaxis,:,:,::-1]
    img = np.tile(img,(5,1,1,1))
    tf.summary.image('Image', img)

    t = tf.constant('Hello, world!')
    tf.summary.text('Text',t)

    writer = tf.summary.FileWriter('./logs_1')
    writer.add_graph(tf.get_default_graph())
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            summary = sess.run(merged)
            writer.add_summary(summary, i)

    writer.close()




if __name__ == '__main__':
    main()