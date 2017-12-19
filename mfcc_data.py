import random
import numpy as np


class MFCC_DATA(object):
    # data, labels: 2d numpy array, every row stands for a sample
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        num = len(data)
        if num != len(labels):
            raise Exception('size not equal for data and labels')
        self.data_size = num
        idx = list(range(num))
        random.shuffle(idx)
        self.idx = np.array(idx)
        self.cur = 0

    # @property
    # def data(self):
    #     return self.data
    #
    # @property
    # def labels(self):
    #     return self.labels
    #
    # @property
    # def idx(self):
    #     return self.idx
    #
    # @property
    # def cur(self):
    #     return self.cur

    def next_batch(self, batch_size):
        iidx = np.array([i % self.data_size for i in range(self.cur, self.cur + batch_size)])
        batch_idx = self.idx[iidx]
        self.cur += batch_size
        self.cur %= self.data_size
        return self.data[batch_idx], self.labels[batch_idx]

    def batch_num(self, batch_size):
        return int(self.data_size / batch_size)

    def shuffle_data(self):
        return self.data[self.idx]

    def shuffle_labels(self):
        return self.labels[self.idx]


class AccQueue:
    def __init__(self, num):
        self.size = num
        self.empties = num
        self.values = np.zeros((num,))
        self.i = 0

    def add(self, acc):
        self.values[self.i] = acc
        self.i = (self.i + 1) % self.size
        self.empties -= 1

    def is_empty(self):
        if self.empties > 0:
            return True
        else:
            return False

    def mean(self):
        return np.average(self.values)


def divide_data(data, labels, continuous_eles, rate1):
    num = len(data)
    if num != len(labels):
        raise Exception('size not equal for data and labels')
    reshape_end = num - num % continuous_eles
    idx1 = np.array(range(num))
    idx2 = idx1[:reshape_end].reshape((-1, continuous_eles))
    np.random.shuffle(idx2)
    idx = np.hstack((idx2.ravel(), idx1[reshape_end:]))
    split = int(num * rate1)
    data1 = data[idx[:split]]
    labels1 = labels[idx[:split]]
    data2 = data[idx[split:]]
    labels2 = labels[idx[split:]]
    return (data1, labels1), (data2, labels2)
