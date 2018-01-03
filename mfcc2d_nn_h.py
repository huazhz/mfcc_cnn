import tensorflow as tf
import numpy as np
import load_data
import data_set
import tempfile
import config_h
import post_process
import os
import sys
import operator
from itertools import accumulate
from functools import reduce
import post_process

os.environ["CUDA_VISIBLE_DEVICES"] = config_h.visible_device


def deep_nn(x, k_prob):
    with tf.name_scope('reshape'):
        x_mfcc = tf.reshape(x, [-1, config_h.mfcc_n, 39, 1])

    with tf.name_scope('conv1'):
        w_c1 = weight_variable(config_h.cnn_kernel1_shape)
        b_c1 = bias_variable(config_h.cnn_kernel1_shape[-1:])
        h_c1 = tf.nn.relu(conv2d(x_mfcc, w_c1) + b_c1)

    with tf.name_scope('pool1'):
        h_p1 = max_pool(h_c1)

    with tf.name_scope('conv2'):
        w_c2 = weight_variable(config_h.cnn_kernel2_shape)
        b_c2 = bias_variable(config_h.cnn_kernel2_shape[-1:])
        h_c2 = tf.nn.relu(conv2d(h_p1, w_c2) + b_c2)

    with tf.name_scope('pool2'):
        h_p2 = max_pool(h_c2)

    with tf.name_scope('fc1'):
        [_, d2, d3, d4] = h_p2.shape.as_list()
        h_p2_flat = tf.reshape(h_p2, [-1, d2 * d3 * d4])
        w_fc1 = weight_variable([d2 * d3 * d4, config_h.fc1_fs])
        b_fc1 = bias_variable([config_h.fc1_fs])
        h_fc1 = tf.nn.relu(tf.matmul(h_p2_flat, w_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        h_fc1_dropout = tf.nn.dropout(h_fc1, k_prob)

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([config_h.fc1_fs, len(config_h.classes)])
        b_fc2 = bias_variable([len(config_h.classes)])
        h_fc2 = tf.matmul(h_fc1_dropout, w_fc2) + b_fc2

    # y_conv = tf.nn.softmax(h_fc2)

    return h_fc2


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=config_h.conv_strides, padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=config_h.maxpool_ksize, strides=config_h.maxpool_strides, padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    log_epoch_interval = config_h.log_epoch_interval
    n_classes = len(config_h.classes)
    loss_weights = tf.constant(config_h.loss_weights, dtype=tf.float32)
    learning_rates = config_h.learning_rates
    restore_file = config_h.restore_file
    restart_epoch_i = config_h.restart_epoch_i
    loop_epoch_nums = config_h.loop_epoch_nums
    persist_checkpoint_interval = config_h.persist_checkpoint_interval
    persist_checkpoint_file = config_h.persist_checkpoint_file

    train_data = np.load(config_h.train_d_npy)
    train_ls = np.load(config_h.train_l_npy)
    mfcc_train = data_set.DataSet(train_data, train_ls)
    vali_data = np.load(config_h.vali_d_npy)
    vali_ls = np.load(config_h.vali_l_npy)
    mfcc_vali = data_set.DataSet(vali_data, vali_ls)
    test_data = np.load(config_h.test_d_npy)
    test_ls = np.load(config_h.test_l_npy)
    mfcc_test = data_set.DataSet(test_data, test_ls)

    x = tf.placeholder(tf.float32, [None, config_h.mfcc_n, 39])
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    k_prob = tf.placeholder(tf.float32)
    y_conv = deep_nn(x, k_prob)
    learning_rate_ph = tf.placeholder(tf.float32)

    with tf.name_scope('loss'):
        # tmp1 = y_ * tf.log(y_conv)
        # print('tmp1 shape', tmp1.shape)
        # tmp2 = tmp1 * loss_weights
        # print('tmp2 shape', tmp2.shape)
        # cross_entroys = -tf.reduce_mean(y_ * tf.log(y_conv) * loss_weights, reduction_indices=[1])
        y_label = y_ * loss_weights
        cross_entroys = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_conv)
        # cross_entroys = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    loss = tf.reduce_mean(cross_entroys)

    with tf.name_scope('adadelta_optimizer'):
        train_step = tf.train.AdadeltaOptimizer(learning_rate_ph).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # graph_location = tempfile.mkdtemp()
    # print('Saving graph to: %s' % graph_location)
    # train_writer = tf.summary.FileWriter(graph_location)
    # train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if config_h.is_train:
            start_i = 0
            end_i = reduce((lambda _a, _b: _a + _b), loop_epoch_nums, 0)
            if config_h.is_restore:
                saver.restore(sess, restore_file)
                start_i = restart_epoch_i
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
            for i in range(start_i, end_i):
                if i % log_epoch_interval == 0:
                    train_acc, train_loss = acc_loss_epoch(x, y_, k_prob, accuracy, loss, mfcc_train, sess)
                    vali_acc, vali_loss = acc_loss_epoch(x, y_, k_prob, accuracy, loss, mfcc_vali, sess)
                    print('epoch %d , train_acc %g , train_loss %g , vali_acc %g, vali_loss %g' % (
                        i, train_acc, train_loss, vali_acc, vali_loss))
                if i % persist_checkpoint_interval == 0:
                    saver.save(sess, persist_checkpoint_file+str(i))
                # train_step = get_train_step(train_steps, loop_epoch_nums, i)
                lr = get_lr(learning_rates, loop_epoch_nums, i)
                # print('learning rate', lr)
                train_epoch(x, y_, k_prob, train_step, learning_rate_ph, lr, mfcc_train, sess)
        else:
            saver.restore(sess, restore_file)
        test_acc, test_loss = acc_loss_epoch(x, y_, k_prob, accuracy, loss, mfcc_test, sess)
        print('test_acc %g , test_loss %g' % (test_acc, test_loss))
        save_result(x, y_, k_prob, y_conv, mfcc_test, sess)


def save_result(x, y_, k_prob, y_conv, d_set, sess):
    batch_size = config_h.batch_size
    gt_pickle = config_h.gt_pickle
    pr_pickle = config_h.pr_pickle
    g_ts = list()
    p_rs = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_y_conv = y_conv.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: 1
        }, session=sess)
        batch_p_r = np.argmax(batch_y_conv, 1)
        batch_g_t = np.argmax(batch_y, 1)
        g_ts += list(batch_g_t)
        p_rs += list(batch_p_r)
    post_process.dump_list(g_ts, gt_pickle)
    post_process.dump_list(p_rs, pr_pickle)


def train_epoch(x, y_, k_prob, train_step, learning_rate_ph, lr, train_set, sess):
    batch_size = config_h.batch_size
    train_k_prob = config_h.train_k_prob
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = train_set.next_batch_fix2(batch_size)
        train_step.run(feed_dict={
            x: batch_x, y_: batch_y, k_prob: train_k_prob, learning_rate_ph: lr
        }, session=sess)


def acc_loss_epoch(x, y_, k_prob, acc_tf, loss_tf, d_set, sess):
    batch_size = config_h.batch_size
    losses = list()
    acces = list()
    weights = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_loss = loss_tf.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: 1
        }, session=sess)
        losses.append(batch_loss)
        batch_acc = acc_tf.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: 1
        }, session=sess)
        acces.append(batch_acc)
        weights.append(len(batch_x))
    epoch_acc = float(np.dot(acces, weights) / np.sum(weights))
    epoch_loss = float(np.dot(losses, weights) / np.sum(weights))
    return epoch_acc, epoch_loss


def get_lr(lrs, train_epoch_nums, current_num):
    acc_train_epoch_nums = accumulate(train_epoch_nums, operator.add)
    for lr, acc_train_epoch_num in zip(lrs, acc_train_epoch_nums):
        if current_num < acc_train_epoch_num:
            return lr
    return lrs[-1]

# def get_train_step(train_steps, train_epoch_nums, current_num):
#     acc_train_epoch_nums = accumulate(train_epoch_nums, operator.add)
#     for train_step, acc_train_epoch_num in zip(train_steps, acc_train_epoch_nums):
#         if current_num < acc_train_epoch_num:
#             return train_step
#     return train_steps[-1]


def print_config(cfg_file='./config_h.py'):
    with open(cfg_file, 'r') as cfg_f:
        for line in cfg_f:
            if '#' in line:
                if 'old' in line:
                    break
                continue
            if '=' in line:
                print(line)


if __name__ == '__main__':
    print_config()
    tf.app.run(main=main, argv=[sys.argv[0]])
    print('id_str', config_h.id_str)
