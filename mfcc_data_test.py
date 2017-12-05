import mfcc_data
import load_data
import time
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
    t1 = time.time()
    data, labels, classes = load_data.load_data_1hot('./trim_labels_all.txt')
    t2 = time.time()
    print(t2 - t1)
    mfcc_train = mfcc_data.MFCC_DATA(data, labels)
    print(len(data))
    batch_size = 200000
    loop = 100
    for _ in range(loop):
        mfcc_train.next_batch(batch_size)
        print(mfcc_train.cur)


if __name__ == '__main__':
    test_mfcc_data2()

