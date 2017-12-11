import mfcc_data
import load_data
import time
import config
import numpy as np


def test_mfcc_data():
    data_raw = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    labels_raw = np.array([3, 7, 11, 15, 19, 23])
    mfcc_train = mfcc_data.MFCC_DATA(data_raw, labels_raw)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    data, labels = mfcc_train.next_batch(1)
    print_dl(data, labels)
    data, labels = mfcc_train.next_batch(2)
    print_dl(data, labels)
    data, labels = mfcc_train.next_batch(3)
    print_dl(data, labels)
    data, labels = mfcc_train.next_batch(4)
    print_dl(data, labels)
    data, labels = mfcc_train.next_batch(5)
    print_dl(data, labels)
    data, labels = mfcc_train.next_batch(7)
    print_dl(data, labels)


def print_dl(data, labels):
    for row, l in zip(data, labels):
        print(row, l)
    print()


def test_mfcc_data2():
    n = 40
    t1 = time.time()
    data, labels = load_data.load_data2d_1hot2('./trim_labels_all.txt', n, config.classes)
    t2 = time.time()
    print(t2 - t1)
    mfcc_train = mfcc_data.MFCC_DATA(data, labels)
    print(len(data))
    batch_size = 20000
    loop = 10
    for _ in range(loop):
        batch = mfcc_train.next_batch(batch_size)
        print(batch[0].shape, batch[1].shape)
        print(mfcc_train.cur)


def test_divide_data():
    data_raw = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
    labels_raw = np.array([3, 7, 11, 15, 19, 23, 27])
    rate1 = 0.67
    (train_data, train_label), (test_data, test_label) = mfcc_data.divide_data(data_raw, labels_raw, 3, rate1)
    print_dl(train_data, train_label)
    print_dl(test_data, test_label)


def test_divide_data2():
    data_raw, labels_raw, classes = load_data.load_data_1hot('./trim_labels_1.txt')
    rate1 = 0.67
    (train_data, train_label), (test_data, test_label) = mfcc_data.divide_data(data_raw, labels_raw, 10, rate1)
    print(len(data_raw))
    print(len(train_data))
    print(len(test_data))
    print(len(train_data) + len(test_data))


if __name__ == '__main__':
    test_mfcc_data2()
    # test_divide_data()
