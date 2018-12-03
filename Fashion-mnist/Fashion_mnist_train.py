import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import Fashion_mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 定义训练得到模型保存的路径和文件名
MODEL_SAVE_PATH = 'model'
MODEL_NAME = 'model.ckpt'

# 训练阶段
def train(fashion_mnist):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, Fashion_mnist_inference.IMAGE_SIZE, Fashion_mnist_inference.IMAGE_SIZE,
                                    Fashion_mnist_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, Fashion_mnist_inference.OUT_NODE], name='y-input')


    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = Fashion_mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, fashion_mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 每训练1000次，保存一次权重文件
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = fashion_mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, Fashion_mnist_inference.IMAGE_SIZE, Fashion_mnist_inference.IMAGE_SIZE,
                                          Fashion_mnist_inference.NUM_CHANNELS))
            _, loss_value, step , Ac = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training"
                      "batch is %g., accuracy = %g." % (step, loss_value, Ac))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    fashion_mnist = input_data.read_data_sets("Fashion-mnist-data", one_hot=True)
    train(fashion_mnist)


if __name__ == '__main__':
    tf.app.run()
