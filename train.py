import tensorflow as tf
import sys
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow.examples.tutorials.mnist import input_data
from print_functions import *
from tf_wrappers import *
import matplotlib.pyplot as plt


def train(generator, discriminator):
    def noise_generator(batch_size, noise_dim):
        ret = []
        for i in range(0, noise_dim):
            ret.append(tf.cast(tf.random_uniform([batch_size, 1], minval=-1., maxval=1.),
                               tf.float32))
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
    #  Discriminator part:
    #
    #       \max_d \sum_i log(d(x_i)) + log(1-d(g(z_i)))
    #       d_loss:= \sum_i -log(d(x_i)) -log(1-d(g(z_i)))
    #       @note: log() function is monoton
    #       \min_d d_loss
    #
    #  Generator part:
    #
    #       \min_g  \sum_i log(1-d(g(z_i))) we want d(g(z)) to be 1, that is, to fool the discriminator
    #       \max_g  \sum_i log(d(g(z_i)))
    #       \min_g  \sum_i -log(d(g(z_i)))
    #       g_loss= \sum_i -log(d(g(z_i)))
    #       @note: this transormation is true in the "argmax" sense, check the attached excel: gan_generator_loss.xlsx/pdf
    #
    #
    d_loss_real_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_d,
                                                                            labels=tf.ones_like(
                                                                                real_d)))
    d_loss_fake_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d,
                                                                            labels=tf.zeros_like(
                                                                                fake_d)))
    d_loss_op = d_loss_real_op + d_loss_fake_op
    g_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d,
                                                                       labels=tf.ones_like(
                                                                           fake_d)))
    # g_loss_op = tf.reduce_mean(tf.log(1-tf.sigmoid(fake_d)))


    tvars = tf.trainable_variables()

    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    d_trainer_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(
        d_loss_op, var_list=d_vars)
    g_trainer_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(
        g_loss_op, var_list=g_vars)

    # tricky test noise columns totally different
    #                   rows only differ in one latent variable
    repeat_noise = int(10)
    noise_mtx = np.tile(
        np.random.uniform(-1, 1, size=[repeat_noise, noise_dim - 1]),
        [repeat_noise, 1])
    latent_mtx = np.arange(-1, 1, 2.0 / repeat_noise)[:repeat_noise]
    latent_mtx = np.tile(latent_mtx.reshape(repeat_noise, 1), (1, repeat_noise)).reshape(
        repeat_noise * repeat_noise, 1)

    tricky_z_batch = np.concatenate([latent_mtx, noise_mtx], axis=1)
    tricky_z_batch = np.concatenate([
        tricky_z_batch,
        np.random.uniform(-1, 1,
                          size=[batch_size - (repeat_noise * repeat_noise), noise_dim])
    ], axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("log", sess.graph)

        # orig dataset
        real_image_batch, real_label_batch = mnist.train.next_batch(batch_size)
        print_img_matrix(10, real_image_batch.reshape([batch_size, 28, 28, 1]), "orig",
                         "0")

        # no train
        z_batch = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
        test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
        print_img_matrix(10, test_imgs, "no_train", "0")

        # train
        loss_for_plot = np.zeros((2, epoch_num))
        for epoch in range(epoch_num):
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=iterations, widgets=widgets)
            pbar.start()

            d_loss_avg = 0.0
            g_loss_avg = 0.0
            for i in range(iterations):
                pbar.update(i)

                real_image_batch, real_label_batch = mnist.train.next_batch(batch_size)
                real_image_batch = np.reshape(real_image_batch, [batch_size, 28, 28, 1])

                _, d_loss = sess.run([d_trainer_op, d_loss_op],
                                     feed_dict={ph_x: real_image_batch})
                _, g_loss = sess.run([g_trainer_op, g_loss_op])

                d_loss_avg += (d_loss / iterations)
                g_loss_avg += (g_loss / iterations)

            print("Epoch %d \n| d_loss= %.5e \n| g_loss= %.5e" % (epoch, d_loss, g_loss))
            loss_for_plot[0, epoch] = d_loss
            loss_for_plot[1, epoch] = g_loss
            sys.stdout.flush()

            test_imgs, g_loss = sess.run([fake_x_disp, g_loss_op],
                                         feed_dict={ph_z: tricky_z_batch})
            print_img_matrix(10, test_imgs, "epoch", str(epoch))

        # test
        z_batch = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
        test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
        print_img_matrix(10, test_imgs, "final", "0")


        plt.plot(loss_for_plot[0, :], 'r-', label='Disc loss')
        plt.plot(loss_for_plot[1, :], 'b-', label='Gen loss')
        plt.legend()
        plt.savefig('pics/losses.png')
