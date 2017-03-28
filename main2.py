import tensorflow as tf
import sys
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow.examples.tutorials.mnist import input_data
from print_functions import *
from tf_wrappers import *


mnist = input_data.read_data_sets("MNIST_data/")


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
        s1, g1 = 4, g_dim * 4
        s2, g2 = 7, g_dim * 2
        s3, g3 = 13, g_dim * 1
        s4, g4 = 28, c_dim

        # assert z size = 100 = 2*2*25
        a0 = tf.reshape(z, [batch_size, 2, 2, 25])
        a0 = tf.nn.relu(a0)

        with tf.variable_scope("deconv1"):
            a1 = deconv_batch_relu(a0, [batch_size, s1, s1, g1], 'SAME')

        with tf.variable_scope("deconv2"):
            a2 = deconv_batch_relu(a1, [batch_size, s2, s2, g2], 'SAME')

        with tf.variable_scope("deconv3"):
            a3 = deconv_batch_relu(a2, [batch_size, s3, s3, g3], 'SAME')

        with tf.variable_scope("deconv4"):
            a4 = deconv_tanh(a3, [batch_size, s4, s4, g4], 'VALID')

    return a4

def generator_v3(z, batch_size, reuse=False):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s1 = 14
        s1_ch = 16
        with tf.variable_scope("fully1"):
            s0 = s1 * s1 * s1_ch
            a0 = fully_batch_relu(z, [100, s0], [s0])

        a1r_temp = tf.reshape(a0, [batch_size, s1, s1, s1_ch])
        a1r = tf.nn.relu(a1r_temp)

        s2 = 28
        with tf.variable_scope("deconv1"):
            a2 = deconv_sigmoid(a1r, [batch_size, s2, s2, 1], 'SAME')

    return a2


def noise_generator(batch_size, noise_dim):
    ret = []
    for i in range(0, noise_dim):
        ret.append(tf.cast(tf.random_uniform([batch_size, 1], minval=-1., maxval=1.), tf.float32))
    return tf.concat(ret, 1)


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

g_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.ones_like(fake_d)))
d_loss_real_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_d, labels=tf.ones_like(real_d)))
d_loss_fake_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.zeros_like(fake_d)))
d_loss_op = d_loss_real_op + d_loss_fake_op


# optimizer
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

for i in d_vars:
    print(i.name)
print("-----------")
for i in g_vars:
    print(i.name)

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
