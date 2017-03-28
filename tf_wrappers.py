import tensorflow as tf

def lrelu(x, leak=0.01, name="lrelu"):
    assert leak <= 1
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


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

    after_norm = tf.contrib.layers.batch_norm(inputs=(conv + biases), center=True, scale=True, is_training=True,
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


def fully_sigmoid(input, kernel_shape, bias_shape):
    return tf.sigmoid(fully(input, kernel_shape, bias_shape))


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


def deconv(input, output_shape, pad):
    weights = tf.get_variable('weights',
                              [4, 4, output_shape[-1], int(input.get_shape()[-1])],
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    after_conv_transpose = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape, strides=[1, 2, 2, 1],
                                                  padding=pad)

    biases = tf.get_variable('biases',
                             [output_shape[-1]],
                             initializer=tf.constant_initializer(0))
    return tf.nn.bias_add(after_conv_transpose, biases)


def deconv_relu(input, output_shape, pad):
    after_deconv = deconv(input, output_shape, pad)
    after_norm = tf.contrib.layers.batch_norm(inputs=after_deconv, center=True, scale=True, is_training=True,
                                              scope="batch_norm")
    return tf.nn.relu(after_norm)


def deconv_batch_relu(input, output_shape, pad):
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
