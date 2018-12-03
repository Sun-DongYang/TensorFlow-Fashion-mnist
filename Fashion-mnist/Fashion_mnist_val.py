import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import Fashion_mnist_inference
import Fashion_mnist_train

# 验证阶段
def evaluate(fashion_mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [fashion_mnist.validation.num_examples, Fashion_mnist_inference.IMAGE_SIZE,
                                        Fashion_mnist_inference.IMAGE_SIZE, Fashion_mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, Fashion_mnist_inference.OUT_NODE], name='y-input')
        xs = fashion_mnist.validation.images
        reshaped_xs = np.reshape(xs, (fashion_mnist.validation.num_examples, Fashion_mnist_inference.IMAGE_SIZE,
                                      Fashion_mnist_inference.IMAGE_SIZE, Fashion_mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: fashion_mnist.validation.labels}
        y = Fashion_mnist_inference.inference(x, 0, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(Fashion_mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # 读取最后一次保存的权重文件来验证网络模型的分类性能
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(Fashion_mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation "
                      "accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return


def main(argv=None):
    fashion_mnist = input_data.read_data_sets(r"Fashion-mnist-data", one_hot=True)
    evaluate(fashion_mnist)


if __name__ == '__main__':
    tf.app.run()