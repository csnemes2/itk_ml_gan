import tensorflow as tf
import sys
from tf_wrappers import *
from train import *


def generator(z, batch_size, reuse=False):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Layer 1 Outshape = batch_size x 3 x 3 x 256
        # Layer 2 Outshape = batch_size x 6 x 6 x 128
        # Layer 3 Outshape = batch_size x 12 x 12 x 64
        # Layer 4 Outshape = batch_size x 28 x 28 x 1

        # z size = 100 = 2*2*25
        a0 = tf.reshape(z, [batch_size, 2, 2, 25])
        a0 = tf.nn.relu(a0)

        s1, g1 = 4, 256
        with tf.variable_scope("deconv1"):
            a1 = deconv_batch_relu(a0, [batch_size, s1, s1, g1], 'SAME')

        s2, g2 = 7, 128
        with tf.variable_scope("deconv2"):
            a2 = deconv_batch_relu(a1, [batch_size, s2, s2, g2], 'SAME')

        s3, g3 = 13, 64
        with tf.variable_scope("deconv3"):
            a3 = deconv_batch_relu(a2, [batch_size, s3, s3, g3], 'SAME')

        s4, g4 = 28, 1
        with tf.variable_scope("deconv4"):
            a4 = deconv_sigmoid(a3, [batch_size, s4, s4, g4], 'VALID')

    return a4


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