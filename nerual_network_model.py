import tensorflow as tf


def deepnn(x, n_class):
    """
    builds the graph for a deep net for classifying emotions
    :param x: input tensor. (N_example, 39)
           n_class: the number of classes
    :return: A tuple (y, keep_prob). y is a tensor of shape (N_example, len(classes)).
    """
    with tf.name_scope('reshape'):
        x_mfcc = tf.reshape(x, [-1, 39, 1, 1])

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([5, 1, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_mfcc, w_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        # n_sample * 20 * 1 * 32
        h_pool1 = max_pool_2x1(h_conv1)

    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5, 1, 1, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        # n_sample * 10 * 1 * 64
        h_pool2 = max_pool_2x1(h_conv2)

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([10*1*64, 256])
        b_fc1 = weight_variable([256])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 10*1*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([256, n_class])
        b_fc2 = bias_variable([n_class])
        y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


