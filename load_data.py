import HTK
import os
import config
import numpy as np


# load data with sentence info
def load_data2(simple_label_file):
    data = list()
    labels = list()
    with open(simple_label_file, 'r') as inf:
        for line in inf:
            eles = line.split()
            if len(eles) == 2:
                mfcc_path = get_mfcc_path(eles[0])
                samples = load_mfcc(mfcc_path)
                data.append(samples)
                labels.append(eles[1])
    return np.array(data), np.array(labels)


def load_data2_np(simple_label_file):
    data = list()
    labels = list()
    with open(simple_label_file, 'r') as inf:
        for line in inf:
            eles = line.split()
            if len(eles) == 2:
                mfcc_path = get_mfcc_path(eles[0])
                samples = load_mfcc_np(mfcc_path)
                data.append(samples)
                labels.append(eles[1])
    return np.array(data), np.array(labels)


def load_mfcc(filename):
    htk = HTK.HTKFile()
    htk.load(filename)
    return htk.data


def load_mfcc_np(file_name):
    return np.array(load_mfcc(file_name))


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


# todo: code from line 140 to line 197 need more test
# load data 2d with shift
def div_sens(sens_data, n, shift):
    results = list()
    len1d = len(sens_data)
    if len1d < n:
        return None
    i = 0
    while i + n <= len1d:
        results.append(sens_data[i:i+n])
        i += shift
    if i - shift + n < len1d:
        results.append(sens_data[-n:])
    return results


# sets, n * (m * 39), a list with n elements, every element is a m * 39 np.array. ls: n
def div_senses(sets, sets_ls, n, shift):
    data = list()
    ls = list()
    for ele, l in zip(sets, sets_ls):
        rs = div_sens(ele, n, shift)
        if rs is None:
            continue
        single_ls = [l] * len(rs)
        data += rs
        ls += single_ls
    return np.array(data), np.array(ls)


# norm
def get_mu_sigma(train_origin_data, train_labels, target_emos):
    m = len(train_origin_data)
    idx = np.zeros((m,), dtype=bool)
    for emo in target_emos:
        idx += (train_labels == emo)
    target_data_origin = train_origin_data[idx]
    target_data_list = list()
    for sentence_mfcc in target_data_origin:
        target_data_list += sentence_mfcc
    target_data = np.array(target_data_list)
    mu = np.average(target_data, axis=0)
    sigma = np.std(target_data, axis=0)
    return mu, sigma


def normalize_set(data, mu, sigma):
    return (data - mu) / sigma


def normalize_origin_set(origin_set, mu, sigma):
    results = list()
    for lists in origin_set:
        result = normalize_set(np.array(lists), mu, sigma)
        results.append(result)
    return results


def norm_train_test_set(train_origin_data, train_labels, test_origin_data, target_emos):
    mu, sigma = get_mu_sigma(train_origin_data, train_labels, target_emos)
    train_norm_data = normalize_origin_set(train_origin_data, mu, sigma)
    test_norm_data = normalize_origin_set(test_origin_data, mu, sigma)
    return train_norm_data, test_norm_data
