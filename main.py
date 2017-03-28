import tensorflow as tf
import sys
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow.examples.tutorials.mnist import input_data
from print_functions import *
from tf_wrappers import *


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


def noise_generator(batch_size, noise_dim):
    ret = []
    for i in range(0, noise_dim):
        ret.append(tf.cast(tf.random_uniform([batch_size, 1], minval=-1., maxval=1.), tf.float32))
    return tf.concat(ret, 1)

mnist = input_data.read_data_sets("MNIST_data/")

noise_dim = 100
batch_size = 128
iterations = 100
epoch_num = 10

with tf.variable_scope("noise"):
    mynoise = tf.random_uniform([batch_size, noise_dim], -1., 1.)

# placeholder.shape[0] = batch size, None means arbitrary
ph_x = tf.placeholder("float", shape=[None, 28, 28, 1])
ph_z = tf.placeholder("float", shape=[None, noise_dim])
noise = noise_generator(batch_size, noise_dim)

# generator
fake_x = generator(noise, batch_size)
fake_d = discriminator(fake_x)

# discriminator
real_d = discriminator(ph_x, reuse=True)

# display
fake_x_disp = generator(ph_z, batch_size, reuse=True)

#
# GAN theory:
#
#  Main:
#       \min_g \max_d \sum_i log(d(x_i)) + log(1-d(g(z_i)))
#
#  That is:
#       \min_g \sum_i log(1-d(g(z_i)))
#       \max_d \sum_i log(d(x_i)) + log(1-d(g(z_i)))
#
#
#  Losses - Using tf.nn.sigmoid_cross_entropy_with_logits():
#
#   tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
#       https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.sigmoid_cross_entropy_with_logits.md
#       x = logits, z = labels
#       z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
#
#   tf.zeros_like = 'zeros' out the vector
#   tf.ones_like = 'ones' out the vector
#
#  Generator part:
#
#       \min_g  \sum_i log(1-d(g(z_i))) we want d(g(z)) to be 1, that is, to fool the discriminator
#       \max_g  \sum_i log(d(g(z_i)))
#       \min_g  \sum_i -log(d(g(z_i)))
#       g_loss= \sum_i -log(d(g(z_i)))
#       @note: this transormation is true in the "argmax" sense, check the attached excel: gan_generator_loss.xlsx/pdf
#
#  Discriminator part:
#
#       \max_d \sum_i log(d(x_i)) + log(1-d(g(z_i)))
#       d_loss:= \sum_i -log(d(x_i)) -log(1-d(g(z_i)))
#       @note: log() function is monoton
#       \min_d d_loss
#
#
g_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.ones_like(fake_d)))
d_loss_real_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_d, labels=tf.ones_like(real_d)))
d_loss_fake_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.zeros_like(fake_d)))
d_loss_op = d_loss_real_op + d_loss_fake_op


tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

d_trainer_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(d_loss_op, var_list=d_vars)
g_trainer_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(g_loss_op, var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("log", sess.graph)

    # orig dataset
    real_image_batch, real_label_batch = mnist.train.next_batch(batch_size)
    print_img_matrix(10, real_image_batch.reshape([batch_size, 28, 28, 1]), "orig", "0")

    # no train
    z_batch = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
    test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
    print_img_matrix(10, test_imgs, "no_train", "0")

    # train
    for epoch in range(epoch_num):
        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=iterations, widgets=widgets)
        pbar.start()
        for i in range(iterations):
            pbar.update(i)

            real_image_batch, real_label_batch = mnist.train.next_batch(batch_size)
            real_image_batch = np.reshape(real_image_batch, [batch_size, 28, 28, 1])

            _, d_loss = sess.run([d_trainer_op, d_loss_op], feed_dict={ph_x: real_image_batch})
            _, g_loss = sess.run([g_trainer_op, g_loss_op])

        print("Epoch %d \n| d_loss= %.5e \n| g_loss= %.5e" % (epoch, d_loss, g_loss))
        sys.stdout.flush()

        repeat_noise = int(10)
        z_batch = np.concatenate([
            np.tile(
                np.random.uniform(-1, 1, size=[repeat_noise, noise_dim]),
                [repeat_noise, 1]
            ),
            np.random.uniform(-1, 1, size=[batch_size - (repeat_noise * repeat_noise), noise_dim])
        ], axis=0)

        test_imgs, g_loss = sess.run([fake_x_disp, g_loss_op], feed_dict={ph_z: z_batch})
        print_img_matrix(10, test_imgs, "epoch", str(epoch))

    # test
    z_batch = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
    test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
    print_img_matrix(10, test_imgs, "final", "0")
