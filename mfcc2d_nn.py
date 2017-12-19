import tensorflow as tf
import numpy as np
import load_data
import mfcc_data
import tempfile
import config
import post_process
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = config.visible_device


def deepnn(x, n_classes):
    with tf.name_scope('reshape'):
        x_mfcc = tf.reshape(x, [-1, config.mfcc_n, 39, 1])

    # [40, 39, 1] to [40, 39, 32]
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable(config.cnn_kernel1_shape)
        b_conv1 = bias_variable(config.cnn_kernel1_shape[-1:])
        h_conv1 = tf.nn.relu(conv2d(x_mfcc, w_conv1) + b_conv1)

    # [40, 39, 32] to [20, 20, 32]
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # [20, 20, 32] to [20, 20, 64]
    with tf.name_scope('conv2'):
        w_conv2 = weight_variable(config.cnn_kernel2_shape)
        b_conv2 = bias_variable(config.cnn_kernel2_shape[-1:])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    # [20, 20, 64] to [10, 10, 64]
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        d3_l = int(((config.mfcc_n + 1) / 2 + 1) / 2)
        flat_l = d3_l * 10 * config.cnn_kernel2_shape[-1]
        w_fc1 = weight_variable([flat_l, config.fc1_fs])
        b_fc1 = bias_variable([config.fc1_fs])
        h_pool2_flat = tf.reshape(h_pool2, [-1, flat_l])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([config.fc1_fs, n_classes])
        b_fc2 = bias_variable([n_classes])
        y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    batch_size = config.batch_size
    # step_num = config.step_num
    print_interval = config.print_interval
    train_data = np.load(config.train_d_npy)
    train_labels = np.load(config.train_l_npy)
    mfcc_train = mfcc_data.MFCC_DATA(train_data, train_labels)
    train_acc_queue = mfcc_data.AccQueue(len(train_data))
    test_data = np.load(config.test_d_npy)
    test_labels = np.load(config.test_l_npy)
    test_acc_queue = mfcc_data.AccQueue(len(test_data))
    mfcc_test = mfcc_data.MFCC_DATA(test_data, test_labels)
    n_classes = len(config.classes)
    x = tf.placeholder(tf.float32, [None, config.mfcc_n, 39])
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    y_conv, keep_prob = deepnn(x, n_classes)

    with tf.name_scope('loss'):
        cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entroy = tf.reduce_mean(cross_entroy)

    with tf.name_scope('adam_optimizer'):
        train_step0 = tf.train.AdadeltaOptimizer(config.learning_rate0).minimize(cross_entroy)
        train_step1 = tf.train.AdadeltaOptimizer(config.learning_rate1).minimize(cross_entroy)
        train_step2 = tf.train.AdadeltaOptimizer(config.learning_rate2).minimize(cross_entroy)
        train_step3 = tf.train.AdadeltaOptimizer(config.learning_rate3).minimize(cross_entroy)

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
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(config.step_num0):
            if i % config.save_checkpoint_interval == 0:
                saver.save(sess, config.checkpoint_file, global_step=i)
            if i % print_interval == 0:
                for j in range(mfcc_train.batch_num(batch_size)):
                    batch_d1, batch_ls1 = mfcc_train.next_batch(batch_size)
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_d1, y_: batch_ls1, keep_prob: 1.0
                    })
                    train_acc_queue.add(train_accuracy)
                print('rate0 step %d, training accuracy %g' % (i, train_acc_queue.mean()))
            batch_data, batch_ls = mfcc_train.next_batch(batch_size)
            train_step0.run(feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 0.5
            })

        for i in range(config.num0, config.num0 + config.step_num1):
            if i % print_interval == 0:
                for j in range(mfcc_train.batch_num(batch_size)):
                    batch_d1, batch_ls1 = mfcc_train.next_batch(batch_size)
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_d1, y_: batch_ls1, keep_prob: 1.0
                    })
                    train_acc_queue.add(train_accuracy)
                print('rate1 step %d, training accuracy %g' % (i, train_acc_queue.mean()))
            batch_data, batch_ls = mfcc_train.next_batch(batch_size)
            train_step1.run(feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 0.5
            })

        for i in range(config.num0 + config.step_num1, config.num0 + config.step_num1 + config.step_num2):
            if i % print_interval == 0:
                for j in range(mfcc_train.batch_num(batch_size)):
                    batch_d1, batch_ls1 = mfcc_train.next_batch(batch_size)
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_d1, y_: batch_ls1, keep_prob: 1.0
                    })
                    train_acc_queue.add(train_accuracy)
                print('rate2 step %d, training accuracy %g' % (i, train_acc_queue.mean()))
            batch_data, batch_ls = mfcc_train.next_batch(batch_size)
            train_step2.run(feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 0.5
            })

        for i in range(config.num0 + config.step_num1 + config.step_num2,
                       config.num0 + config.step_num1 + config.step_num2 + config.step_num3):
            if i % print_interval == 0:
                for j in range(mfcc_train.batch_num(batch_size)):
                    batch_d1, batch_ls1 = mfcc_train.next_batch(batch_size)
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_d1, y_: batch_ls1, keep_prob: 1.0
                    })
                    train_acc_queue.add(train_accuracy)
                print('rate3 step %d, training accuracy %g' % (i, train_acc_queue.mean()))
            batch_data, batch_ls = mfcc_train.next_batch(batch_size)
            train_step3.run(feed_dict={
                x: batch_data, y_: batch_ls, keep_prob: 0.5
            })

        test_accuracy = 0
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
        print("test accuracy %g" % test_accuracy)
        print('Saving graph to: %s' % graph_location)
        print('Saving graph to: %s' % graph_location)
        # post_process.dump_list(classes, config.classes_pickle)
        post_process.dump_list(ground_truth, config.gt_pickle)
        post_process.dump_list(predict_results, config.pr_pickle)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
