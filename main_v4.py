import tensorflow as tf
import sys
from tf_wrappers import *
from train import *

def generator(z, batch_size, reuse=False, img_size=28):
    """
    A Generator inspired by InfoGAN's generator: https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
    """
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("fully1"):
            a0 = fully_batch_relu(z, [100, 1024], [1024])

        s1 = int(img_size / 4)
        with tf.variable_scope("fully2"):
            a1 = fully_batch_relu(a0, [1024, s1 * s1 * 128], [s1 * s1 * 128])

        a1r = tf.reshape(a1, [batch_size, s1, s1, 128])

        s2 = int(img_size / 2)
        with tf.variable_scope("deconv2"):
            a2 = deconv_batch_relu(a1r, [batch_size, s2, s2, 64], 'SAME')

        with tf.variable_scope("deconv3"):
            a3 = deconv_sigmoid(a2, [batch_size, img_size, img_size, 1], 'SAME')

    return a3


def discriminator(x_image, reuse=False):

    with tf.variable_scope("d_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv1"):
            ret = conv_relu_pool(x_image, [5, 5, 1, 8], [8])

        with tf.variable_scope("conv2"):
            ret2 = conv_relu_pool(ret, [5, 5, 8, 16], [16])

        ret3 = tf.reshape(ret2, [-1, 7 * 7 * 16])

        with tf.variable_scope("fully1"):
            ret4 = fully_relu(ret3, [7 * 7 * 16, 32], [32])

        with tf.variable_scope("fully2"):
            ret5 = fully(ret4, [32, 1], [1])

    return ret5

train(generator, discriminator)