import tensorflow as tf
import sys
from tf_wrappers import *
from train import *


def generator(z, batch_size, reuse=False):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s1 = 14
        s1_ch = 64
        with tf.variable_scope("fully1"):
            s0 = s1 * s1 * s1_ch
            a0 = fully_batch_relu(z, [100, s0], [s0])

        a1r_temp = tf.reshape(a0, [batch_size, s1, s1, s1_ch])
        a1r = tf.nn.relu(a1r_temp)

        s2 = 28
        with tf.variable_scope("deconv1"):
            a2 = deconv_sigmoid(a1r, [batch_size, s2, s2, 1], 'SAME')

    return a2


def discriminator(x_image, reuse=False):
    """
    A Generator inspired by InfoGAN's generator: https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
    """
    with tf.variable_scope("d_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv1"):
            ret = conv_stride2_leaky(x_image, [4, 4, 1, 64], [64])

        with tf.variable_scope("conv2"):
            ret2 = conv_stride2_norm_leaky(ret, [4, 4, 64, 128], [128])

        ret3 = tf.reshape(ret2, [-1, 7 * 7 * 128])

        with tf.variable_scope("fully1"):
            ret4 = fully_batch_leaky(ret3, [7 * 7 * 128, 1024], [1024])

        with tf.variable_scope("fully2"):
            ret5 = fully(ret4, [1024, 1], [1])

        return ret5

train(generator, discriminator)