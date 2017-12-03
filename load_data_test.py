import load_data
import numpy as np
from collections import Counter


def test_get_mfcc_path():
    mfcc_key = 'Ses01M_impro06_F002'
    result = load_data.get_mfcc_path(mfcc_key)
    with open(result, 'rb') as f:
        print(f.read())


def test_load_data():
    simple_labels_file = 'trim_labels_1.txt'
    data, labels = load_data.load_data(simple_labels_file)
    print(data.shape)
    print(labels.shape)
    classes = load_data.get_classes(labels)
    new_labels = load_data.get_one_hot_labels(labels, classes)
    a = np.sum(new_labels, axis=0)
    b = Counter(labels)
    print(classes)
    print(a)
    print(b)
    # print(classes)
    # for i in range(len(data)):
    #     print(data[i], labels[i])
    # for row in data:
    #     print(row)
    # print(type(labels))
    # for label in labels:
    #     print(type(label))


def test_load_data_1hot():
    simple_labels_file = 'trim_labels_1.txt'
    data, labels, classes = load_data.load_data_1hot(simple_labels_file)
    a = np.sum(labels, axis=0)
    print(classes)
    print(a)


def test_get_classes_idx():
    classes = {'exc', 'ang', 'fru', 'sad', 'neu', 'hap'}
    d = load_data.get_classes_idx(classes)
    for i, v in d:
        print(i, v)


if __name__ == '__main__':
    test_load_data_1hot()
    # test_get_mfcc_path()
    # test_load_data()
    # test_get_classes_idx()