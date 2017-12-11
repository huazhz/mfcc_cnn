import HTK
import os
import config
import numpy as np


def load_mfcc(filename):
    htk = HTK.HTKFile()
    htk.load(filename)
    return htk.data


def get_mfcc_path(mfcc_key):
    suffix = '.mfcc'
    fold1 = 'mfcc_iemocap'
    eles = mfcc_key.split('_')
    fold2 = '_'.join(eles[:-1])
    file_name = mfcc_key + suffix
    return os.path.join(config.data_path, fold1, fold2, file_name)


def load_data(simple_label_file):
    data = list()
    labels = list()
    with open(simple_label_file, 'r') as inf:
        for line in inf:
            eles = line.split()
            if len(eles) == 2:
                mfcc_path = get_mfcc_path(eles[0])
                samples = load_mfcc(mfcc_path)
                nSample = len(samples)
                sample_labels = [eles[1]] * nSample
                data += samples
                labels += sample_labels
    return np.array(data), np.array(labels)


def load_data_2d(simple_label_file, n):
    data = list()
    labels = list()
    with open(simple_label_file, 'r') as inf:
        for line in inf:
            eles = line.split()
            if len(eles) == 2:
                mfcc_path = get_mfcc_path(eles[0])
                samples_1d = load_mfcc(mfcc_path)
                samples_2d = samples1dto2d(samples_1d, n)
                sample_labels = [eles[1]] * len(samples_2d)
                data += samples_2d
                labels += sample_labels
    return np.array(data), np.array(labels)


def samples1dto2d(samples1d, n):
    results = list()
    len_1d = len(samples1d)
    i = 0
    while i + n < len_1d:
        results.append(samples1d[i:i+n])
        i += n
    if i < len_1d - 1:
        results.append(samples1d[-n:])
    return results


# {'exc', 'ang', 'fru', 'sad', 'neu', 'hap'}
def get_classes(labels):
    return set(labels)


def get_classes_idx(classes):
    return zip(classes, range(len(classes)))


def get_one_hot_labels(labels, classes):
    classes_num = len(classes)
    samples_num = len(labels)
    new_labels = np.zeros((samples_num, classes_num))
    for i, c in zip(range(classes_num), classes):
        idx = labels == c
        new_labels[:, i][idx] = 1
    return new_labels


def load_data_1hot(simple_label_file):
    data, labels_raw = load_data(simple_label_file)
    classes = sorted(get_classes(labels_raw))
    labels = get_one_hot_labels(labels_raw, classes)
    return data, labels, classes


def load_data2d_1hot(sample_labels_file, n):
    data, labels_raw = load_data_2d(sample_labels_file, n)
    classes = sorted(get_classes(labels_raw))
    labels = get_one_hot_labels(labels_raw, classes)
    return data, labels, classes


def load_data2d_1hot2(sample_labels_file, n, classes):
    data, labels_raw = load_data_2d(sample_labels_file, n)
    labels = get_one_hot_labels(labels_raw, classes)
    return data, labels
