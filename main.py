import tensorflow as tf
import numpy as np
import png
import sys
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


# This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.005, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def print_img_row(dim, x, name="temp", iter="0"):
    img_size = np.shape(x)[1]
    img_collection = np.empty([img_size, img_size * dim])
    for i in range(dim):
        img_collection[:, i * img_size:(i + 1) * img_size] = np.reshape(x[i, :, :, :], [img_size, img_size]) * 255

    png.from_array(img_collection.astype(np.uint8), 'L').save("pics/" + name + "_" + str(iter) + ".png")


def print_img_matrix(dim, x, name="temp", iter="0"):
    img_size = np.shape(x)[1]
    img_collection = np.empty([img_size*dim, img_size * dim])
    for idx, image in enumerate(x):
        if idx >= dim*dim:
            break
        i = idx % dim
        j = idx // dim
        img_collection[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = np.reshape(image, [img_size, img_size]) * 255

    png.from_array(img_collection.astype(np.uint8), 'L').save("pics/" + name + "_" + str(iter) + ".png")


def conv_relu_pool(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    rel = tf.nn.relu(conv + biases)
    return tf.nn.avg_pool(rel, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_leaky_pool(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    rel = lrelu(conv + biases)
    return tf.nn.avg_pool(rel, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_stride2_leaky(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 2, 2, 1], padding='SAME')
    rel = lrelu(conv + biases)
    return rel


def conv_stride2_norm_leaky(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 2, 2, 1], padding='SAME')

    after_norm = tf.contrib.layers.batch_norm(inputs=(conv+biases), center=True, scale=True, is_training=True,
                                              scope="batch_norm")
    rel = lrelu(after_norm)
    return rel


def fully(input, kernel_shape, bias_shape):
    weights = tf.get_variable('weights', kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0))
    return tf.matmul(input, weights) + biases


def fully_relu(input, kernel_shape, bias_shape):
    return tf.nn.relu(fully(input, kernel_shape, bias_shape))


def fully_leaky(input, kernel_shape, bias_shape):
    return lrelu(fully(input, kernel_shape, bias_shape))


def fully_batch_relu(input, kernel_shape, bias_shape):
    after_fully = fully(input, kernel_shape, bias_shape)
    after_norm = tf.contrib.layers.batch_norm(inputs=after_fully, center=True, scale=True, is_training=True,
                                              scope="batch_norm")
    return tf.nn.relu(after_norm)


def fully_batch_leaky(input, kernel_shape, bias_shape):
    after_fully = fully(input, kernel_shape, bias_shape)
    after_norm = tf.contrib.layers.batch_norm(inputs=after_fully, center=True, scale=True, is_training=True,
                                              scope="batch_norm")
    return lrelu(after_norm)


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

# if network_type == "mnist":
#     with tf.variable_scope("d_net"):
#         shared_template = \
#             (pt.template("input").
#              reshape([-1] + list(image_shape)).
#              custom_conv2d(64, k_h=4, k_w=4).
#              apply(leaky_rectify).
#              custom_conv2d(128, k_h=4, k_w=4).
#              conv_batch_norm().
#              apply(leaky_rectify).
#              custom_fully_connected(1024).
#              fc_batch_norm().
#              apply(leaky_rectify))
#         self.discriminator_template = shared_template.custom_fully_connected(1)

def discriminator_v2(x_image, reuse=False):
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

def deconv(input, output_shape, pad):
    weights = tf.get_variable('weights',
                              [4, 4, output_shape[-1], int(input.get_shape()[-1])],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    after_conv_transpose = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape, strides=[1, 2, 2, 1], padding=pad)

    biases = tf.get_variable('biases',
                             [output_shape[-1]],
                             initializer=tf.constant_initializer(0))
    return tf.nn.bias_add(after_conv_transpose, biases)


def deconv_relu(input, output_shape, pad):
    after_deconv = deconv(input, output_shape, pad)
    after_norm = tf.contrib.layers.batch_norm(inputs=after_deconv, center=True, scale=True, is_training=True,
                                              scope="batch_norm")
    return tf.nn.relu(after_norm)


def deconv_tanh(input, output_shape, pad):
    after_deconv = deconv(input, output_shape, pad)
    return tf.nn.tanh(after_deconv)

def deconv_sigmoid(input, output_shape, pad):
    after_deconv = deconv(input, output_shape, pad)
    return tf.nn.sigmoid(after_deconv)

def generator(z, batch_size, reuse=False):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        g_dim = 64  # Number of filters of first layer of generator
        c_dim = 1  # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)

        # Dimensions of 1. deconv matrix = batch_size x 3 x 3 x 256
        # Dimensions of 2. deconv matrix = batch_size x 6 x 6 x 128
        # Dimensions of 3. deconv matrix = batch_size x 12 x 12 x 64
        # Dimensions of 4. deconv matrix = batch_size x 28 x 28 x 1
        s1, g1 = 3, g_dim * 4
        s2, g2 = 6, g_dim * 2
        s3, g3 = 12, g_dim * 1
        s4, g4 = 28, c_dim

        # assert z size = 100 = 2*2*25
        a0 = tf.reshape(z, [batch_size, 2, 2, 25])
        a0 = tf.nn.relu(a0)

        with tf.variable_scope("deconv1"):
            a1 = deconv_relu(a0, [batch_size, s1, s1, g1], 'SAME')

        with tf.variable_scope("deconv2"):
            a2 = deconv_relu(a1, [batch_size, s2, s2, g2], 'SAME')

        with tf.variable_scope("deconv3"):
            a3 = deconv_relu(a2, [batch_size, s3, s3, g3], 'SAME')

        with tf.variable_scope("deconv4"):
            a4 = deconv_tanh(a3, [batch_size, s4, s4, g4], 'VALID')

    return a4


#with tf.variable_scope("g_net"):
#    self.generator_template = \
#        (pt.template("input").
#         custom_fully_connected(1024).
#          fc_batch_norm().
#          apply(tf.nn.relu).
#          custom_fully_connected(image_size / 4 * image_size / 4 * 128).
#          fc_batch_norm().
#          apply(tf.nn.relu).
#          reshape([-1, image_size / 4, image_size / 4, 128]).
#          custom_deconv2d([0, image_size / 2, image_size / 2, 64], k_h=4, k_w=4).
#          conv_batch_norm().
#          apply(tf.nn.relu).
#          custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
#          flatten())

def generator_v2(z, batch_size, reuse=False, img_size=28):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("fully1"):
            a0 = fully_batch_relu(z,[100,1024],[1024])

        s1 = int(img_size / 4)
        with tf.variable_scope("fully2"):
            a1 = fully_batch_relu(a0,[1024,s1*s1*128],[s1*s1*128])

        a1r_temp = tf.reshape(a1, [batch_size, s1, s1, 128])
        a1r = tf.nn.relu(a1r_temp)

        s2 = int(img_size/2)
        with tf.variable_scope("deconv2"):
            a2 = deconv_relu(a1r, [batch_size, s2, s2, 64], 'SAME')

        with tf.variable_scope("deconv3"):
            a3 = deconv_sigmoid(a2, [batch_size, img_size, img_size, 1], 'SAME')

    return a3


def generator_v3(z, batch_size, reuse=False):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s1=14
        s1_ch=16
        with tf.variable_scope("fully1"):
            s0 = s1*s1*s1_ch
            a0 = fully_batch_relu(z,[100,s0],[s0])

        a1r_temp = tf.reshape(a0, [batch_size, s1, s1, s1_ch])
        a1r = tf.nn.relu(a1r_temp)

        s2 = 28
        with tf.variable_scope("deconv1"):
            a2 = deconv_sigmoid(a1r, [batch_size, s2, s2, 1], 'SAME')

    return a2

noise_dim = 100
batch_size = 128
iterations = 500
epoch_num = 10

# placeholder.shape[0] = batch size, None means arbitrary
ph_x = tf.placeholder("float", shape=[None, 28, 28, 1])
ph_z = tf.placeholder("float", shape=[None, noise_dim])

# generator
fake_x = generator_v2(ph_z, batch_size)
fake_d = discriminator(fake_x)

# discriminator
real_d = discriminator(ph_x, reuse=True)

#
# GAN theory:
#
#   Main:
#       \min_g \max_d \sum_i log(d(x_i)) + log(1-d(g(z_i)))
#
#   That is:
#       \min_g \sum_i log(1-d(g(z_i)))
#       \max_d \sum_i log(d(x_i)) + log(1-d(g(z_i)))
#

# Basically all three loss implementations are the same

#
# Losses - Using tf.nn.sigmoid_cross_entropy_with_logits():
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
# TF losses:
#   tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
#       https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.sigmoid_cross_entropy_with_logits.md
#       x = logits, z = labels
#       z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
#
#   tf.zeros_like = 'zeros' out the vector
#   tf.ones_like = 'ones' out the vector
#
g_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.ones_like(fake_d)))
d_loss_real_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_d, labels=tf.ones_like(real_d)))
d_loss_fake_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.zeros_like(fake_d)))
d_loss_op = d_loss_real_op + d_loss_fake_op

# Losses  - from https://github.com/awjuliani/TF-Tutorials/blob/master/DCGAN.ipynb
#  @remember: sigmoid is already in the discriminator
#
#  Generator part:
#       g_loss_op = -tf.reduce_mean(tf.log(Dg))
#
#  Discriminator part:
#       d_loss_op = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg))

# Losses  - from InfoGAN paper
#  @remember: sigmoid is already in the discriminator
#
#  Generator part:
#       generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))
#
#  Discriminator part:
#       discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))



# optimizer
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]
d_trainer_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(d_loss_op, var_list=d_vars)
g_trainer_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(g_loss_op, var_list=g_vars)

for i in d_vars:
    print(str(i))
print()
for i in g_vars:
    print(str(i))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("log", sess.graph)

    # orig dataset
    real_image_batch, real_label_batch = mnist.train.next_batch(batch_size)
    print_img_matrix(10, real_image_batch.reshape([batch_size, 28, 28, 1]), "orig", "0")

    # no train
    z_batch = np.random.normal(-1, 1, size=[batch_size, noise_dim])
    test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
    print_img_row(10, test_imgs, "no_train", "0")
    z_batch = np.random.normal(-1, 1, size=[batch_size, noise_dim])
    test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
    print_img_row(10, test_imgs, "no_train", "1")

    # train
    for epoch in range(epoch_num):
        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=iterations, widgets=widgets)
        pbar.start()
        for i in range(iterations):
            pbar.update(i)
            z_batch = np.random.normal(-1, 1, size=[batch_size, noise_dim])
            real_image_batch, real_label_batch = mnist.train.next_batch(batch_size)
            real_image_batch = np.reshape(real_image_batch, [batch_size, 28, 28, 1])
            _, d_loss, xxx, yyy = sess.run([d_trainer_op, d_loss_op, fake_x, fake_d],
                                           feed_dict={ph_z: z_batch, ph_x: real_image_batch})
            #print("min fake x:",np.min(xxx))
            #print("min real x:",np.min(real_image_batch))

            _, g_loss, fakd = sess.run([g_trainer_op, g_loss_op, fake_d], feed_dict={ph_z: z_batch})
            # print("g")
            # print(g_loss)
            # print(np.transpose(fakd))
            # print("Epoch %d \n| d_loss= %.5e \n| g_loss= %.5e" % (epoch, d_loss, g_loss))
        print("Epoch %d \n| d_loss= %.5e \n| g_loss= %.5e" % (epoch, d_loss, g_loss))
        sys.stdout.flush()

        z_batch = np.random.normal(-1, 1, size=[batch_size, noise_dim])
        test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
        print_img_matrix(5, test_imgs, "during_train", str(epoch))

    # test
    z_batch = np.random.normal(-1, 1, size=[batch_size, noise_dim])
    test_imgs, g_loss = sess.run([fake_x, g_loss_op], feed_dict={ph_z: z_batch})
    print_img_row(10, test_imgs, "after_train", "0")
