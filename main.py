import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


def print_img_matrix(dim,x_train):
    for i in range(dim):
        print(np.shape(x_train[i]))
    return

def conv_relu_pool(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    rel = tf.nn.relu(conv + biases)
    return tf.nn.avg_pool(rel, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fully(input, kernel_shape, bias_shape):
    weights = tf.get_variable('weights', kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0))
    return tf.matmul(input, weights) + biases


def fully_relu(input, kernel_shape, bias_shape):
    return tf.nn.relu(fully(input, kernel_shape, bias_shape))

def discriminator(x_image, reuse=False):

    with tf.variable_scope("d_"):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv1"):
            ret = conv_relu_pool(x_image,[5, 5, 1, 8],[8])

        with tf.variable_scope("conv2"):
            ret2 = conv_relu_pool(ret,[5, 5, 8, 16],[16])

        ret3 = tf.reshape(ret2, [-1, 7 * 7 * 16])

        with tf.variable_scope("fully1"):
            ret4 = fully_relu(ret3,[7 * 7 * 16, 32],[32])

        with tf.variable_scope("fully2"):
            ret5 = fully_relu(ret4,[32, 1],[1])

        return ret5


def deconv(input,output_shape,pad):

    W_conv1 = tf.get_variable('g_wconv1',
                            [5, 5, output_shape[-1],int(input.get_shape()[-1])],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv1 = tf.get_variable('g_bconv1', [output_shape[-1]], initializer=tf.constant_initializer(.1))
    return tf.nn.conv2d_transpose(input, W_conv1, output_shape=output_shape, strides=[1, 2, 2, 1], padding=pad)


def deconv_relu(input,output_shape,pad):
    H_conv1 = deconv(input,output_shape,pad)
    H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
    return tf.nn.relu(H_conv1)

def deconv_tanh(input,output_shape,pad):
    H_conv1 = deconv(input,output_shape,pad)
    return tf.nn.tanh(H_conv1)

def generator(z, batch_size, reuse=False):

    with tf.variable_scope("d_"):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        g_dim = 64  # Number of filters of first layer of generator
        c_dim = 1   # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
        s = 28      # Output size of the image
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int( s / 16)

        a0 = tf.reshape(z, [batch_size, 2, 2, 25])
        a0 = tf.nn.relu(a0)


        with tf.variable_scope("deconv1"):
            # Dimensions of H_conv1 = batch_size x 3 x 3 x 256
            a1 = deconv_relu(a0,[batch_size, s8, s8, g_dim * 4],'SAME')

        with tf.variable_scope("deconv2"):
            # Dimensions of H_conv1 = batch_size x 6 x 6 x 128
            a2 = deconv_relu(a1, [batch_size, s4 - 1, s4 - 1, g_dim * 2],'SAME')

        with tf.variable_scope("deconv3"):
            # Dimensions of H_conv1 = batch_size x 12 x 12 x 64
            a3= deconv_relu(a2, [batch_size, s2 - 2, s2 - 2, g_dim * 1],'SAME')

        with tf.variable_scope("deconv4"):
            # Dimensions of H_conv1 = batch_size x 28 x 28 x 1
            a4= deconv_tanh(a3, [batch_size, s, s, c_dim],'VALID')

    return a4

# Build graph: discriminator
ph_x =  tf.placeholder("float", shape = [None,28,28,1])
d = discriminator(ph_x)

ph_z =  tf.placeholder("float", shape = [None,100])
fake_x = generator(ph_z,64)

x_train = mnist.train.images[:55000,:]
print("Shape of the training set=%s" % str(x_train.shape))


# Lets print out 5x5 images from the training set
print_img_matrix(5,x_train)




# tensorboard log missing