import tensorflow as tf
import load_data
import mfcc_data
import tempfile
import sys
import config
import post_process
import os

os.environ["CUDA_VISIBLE_DEVICES"] = config.visible_device


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
        w_conv2 = weight_variable([5, 1, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        # n_sample * 10 * 1 * 64
        h_pool2 = max_pool_2x1(h_conv2)

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([10 * 1 * 64, 256])
        b_fc1 = weight_variable([256])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 1 * 64])
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


def main(_):
    batch_size = config.batch_size
    step_num = config.step_num
    print_interval = config.print_interval
    data_raw, labels_raw, classes = load_data.load_data_1hot(config.label_file)
    continue_eles = config.continue_eles
    rate1 = config.train_rate
    (train_set, train_labels), (test_set, test_labels) = mfcc_data.divide_data(data_raw, labels_raw, continue_eles,
                                                                               rate1)
    mfcc_train = mfcc_data.MFCC_DATA(train_set, train_labels)
    mfcc_test = mfcc_data.MFCC_DATA(test_set, test_labels)

    x = tf.placeholder(tf.float32, [None, 39])
    y_ = tf.placeholder(tf.float32, [None, len(classes)])

    y_conv, keep_prob = deepnn(x, len(classes))

    with tf.name_scope('loss'):
        cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entroy = tf.reduce_mean(cross_entroy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdadeltaOptimizer(config.learning_rate).minimize(cross_entroy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('result'):
        g_truth = tf.argmax(y_, 1)
        p_rs = tf.argmax(y_conv, 1)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(step_num):
            batch_data, batch_ls = mfcc_train.next_batch(batch_size)
            if i % print_interval == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_data, y_: batch_ls, keep_prob: 1.0
                })
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 0.5
            })

        # evaluate using test set
        test_accuracy = 0
        # print(mfcc_test.batch_num(batch_size))
        ground_truth = list()
        predict_results = list()
        for _ in range(mfcc_test.batch_num(batch_size)):
            batch_data, batch_ls = mfcc_test.next_batch(batch_size)
            test_accuracy += accuracy.eval(feed_dict={x: batch_data, y_: batch_ls, keep_prob: 1.0})
            ground_truth_batch = sess.run(g_truth, feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 1.0
            })
            predict_results_batch = sess.run(p_rs, feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 1.0
            })
            ground_truth += list(ground_truth_batch)
            predict_results += list(predict_results_batch)
        test_batch_num = mfcc_test.batch_num(batch_size)
        if test_batch_num == 0:
            test_batch_num = 1
        test_accuracy /= test_batch_num
        # print(ground_truth)
        # print(predict_results)
        print("test accuracy %g" % test_accuracy)
        print('Saving graph to: %s' % graph_location)
        post_process.dump_list(classes, config.classes_pickle)
        post_process.dump_list(ground_truth, config.gt_pickle)
        post_process.dump_list(predict_results, config.pr_pickle)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
    # main(None)
