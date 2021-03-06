import load_data
import numpy as np
import config
import time
from collections import Counter

train_data_origin_f = 'train_data_origin.npy'
test_data_origin_f = 'test_data_origin.npy'
train_labels_f = 'train_labels.npy'
test_labels_f = 'test_labels.npy'

train_data_pre = 'train_data_'
train_l_pre = 'train_l_'
test_data_pre = 'test_data_'
test_l_pre = 'test_l_'


def test():
    test_l_f = './fil_l5.txt'
    test_data_origin, test_ls_origin, test_sens_ids_origin = load_data.load_data2_with_sentence(
        test_l_f)


def get_test_data_with_sentences():
    train_l_f = './fil_l123.txt'
    test_l_f = './fil_l5.txt'
    classes = config.classes
    train_data_origin, train_ls_origin = load_data.load_data2(train_l_f)
    test_data_origin, test_ls_origin, test_sens_ids_origin = load_data.load_data2_with_sentence(
        test_l_f)
    target_emos = ['neu']
    n = 200
    shift = 40
    mu, sigma = load_data.get_mu_sigma(train_data_origin, train_ls_origin, target_emos)
    # train_norm = load_data.normalize_origin_set(train_data_origin, mu, sigma)
    test_norm = load_data.normalize_origin_set(test_data_origin, mu, sigma)
    test_data, test_ls, test_sens_ids = load_data.div_senses_with_sentences(test_norm,
                                                                            test_ls_origin,
                                                                            test_sens_ids_origin, n,
                                                                            shift)
    test_ls_1hot = load_data.get_one_hot_labels(test_ls, classes)
    test_data_npy_f = 'data_5_sens_n' + str(n) + '_s' + str(shift) + '.npy'
    test_ls_1hot_npy_f = 'ls_1hot_5_sens_n' + str(n) + '_s' + str(shift) + '.npy'
    test_sens_npy_f = 'sens_5_sens_n' + str(n) + '_s' + str(shift) + '.npy'
    np.save(test_data_npy_f, test_data)
    np.save(test_ls_1hot_npy_f, test_ls_1hot)
    np.save(test_sens_npy_f, test_sens_ids)


def get_train_develop_train_data():
    train_l_f = './fil_l123.txt'
    validation_l_f = './fil_l4.txt'
    test_l_f = './fil_l5.txt'
    classes = config.classes
    train_data_origin, train_ls_origin = load_data.load_data2(train_l_f)
    vali_data_origin, vali_ls_origin = load_data.load_data2(validation_l_f)
    test_data_origin, test_ls_origin = load_data.load_data2(test_l_f)
    target_emos = ['neu']
    n = 200
    shift = 40
    mu, sigma = load_data.get_mu_sigma(train_data_origin, train_ls_origin, target_emos)
    train_norm = load_data.normalize_origin_set(train_data_origin, mu, sigma)
    vali_norm = load_data.normalize_origin_set(vali_data_origin, mu, sigma)
    test_norm = load_data.normalize_origin_set(test_data_origin, mu, sigma)
    train_data, train_ls = load_data.div_senses(train_norm, train_ls_origin, n, shift)
    vali_data, vali_ls = load_data.div_senses(vali_norm, vali_ls_origin, n, shift)
    test_data, test_ls = load_data.div_senses(test_norm, test_ls_origin, n, shift)
    train_ls_1hot = load_data.get_one_hot_labels(train_ls, classes)
    vali_ls_1hot = load_data.get_one_hot_labels(vali_ls, classes)
    test_ls_1hot = load_data.get_one_hot_labels(test_ls, classes)
    train_data_npy_f = 'data_123_n' + str(n) + '_s' + str(shift) + '.npy'
    train_ls_1hot_npy_f = 'ls_1hot_123_n' + str(n) + '_s' + str(shift) + '.npy'
    vali_data_npy_f = 'data_4_n' + str(n) + '_s' + str(shift) + '.npy'
    vali_ls_1hot_npy_f = 'ls_1hot_4_n' + str(n) + '_s' + str(shift) + '.npy'
    test_data_npy_f = 'data_5_n' + str(n) + '_s' + str(shift) + '.npy'
    test_ls_1hot_npy_f = 'ls_1hot_5_n' + str(n) + '_s' + str(shift) + '.npy'
    np.save(train_data_npy_f, train_data)
    np.save(train_ls_1hot_npy_f, train_ls_1hot)
    np.save(vali_data_npy_f, vali_data)
    np.save(vali_ls_1hot_npy_f, vali_ls_1hot)
    np.save(test_data_npy_f, test_data)
    np.save(test_ls_1hot_npy_f, test_ls_1hot)


def test_norm_and_div():
    target_emos = ['neu']
    classes = ['neu', 'ang', 'hap', 'sad']
    n = 200
    shift = 40
    train_origin = np.load(train_data_origin_f)
    test_origin = np.load(test_data_origin_f)
    train_labels = np.load(train_labels_f)
    test_labels = np.load(test_labels_f)
    train_norm, test_norm = load_data.norm_train_test_set(train_origin, train_labels, test_origin,
                                                          target_emos)
    train_data, train_ls = load_data.div_senses(train_norm, train_labels, n, shift)
    test_data, test_ls = load_data.div_senses(test_norm, test_labels, n, shift)
    train_ls_1hot = load_data.get_one_hot_labels(train_ls, classes)
    test_ls_1hot = load_data.get_one_hot_labels(test_ls, classes)
    print(train_data.shape, train_ls_1hot.shape)
    print(test_data.shape, test_ls_1hot.shape)
    train_data_filename = train_data_pre + 'n' + str(n) + '_s' + str(shift)
    train_l_filename = train_l_pre + 'n' + str(n) + '_s' + str(shift) + '_1hot'
    test_data_filename = test_data_pre + 'n' + str(n) + '_s' + str(shift)
    test_l_filename = test_l_pre + 'n' + str(n) + '_s' + str(shift) + '_1hot'
    np.save(train_data_filename, train_data)
    np.save(train_l_filename, train_ls_1hot)
    np.save(test_data_filename, test_data)
    np.save(test_l_filename, test_ls_1hot)


def test_load_data2():
    train_l_f = './fil_l1234.txt'
    test_l_f = './fil_l5.txt'
    train_data, train_ls = load_data.load_data2(train_l_f)
    test_data, test_ls = load_data.load_data2(test_l_f)
    np.save('train_data_origin', train_data)
    np.save('test_data_origin', test_data)
    np.save('train_labels', train_ls)
    np.save('test_labels', test_ls)
    print(train_data.shape, train_ls.shape)
    print(test_data.shape, test_ls.shape)
    i = 5
    print(train_ls[5], test_ls[5])


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


def test_load_data2d_1hot():
    t1 = time.time()
    trim_labels_file1 = 'trim_labels_1234.txt'
    n = 40
    data1, labels1 = load_data.load_data2d_1hot2(trim_labels_file1, n, config.classes)
    a1 = np.sum(labels1, axis=0)
    print(len(data1))
    print(a1)
    # print(classes1)
    trim_labels_file2 = 'trim_labels_5.txt'
    data2, labels2 = load_data.load_data2d_1hot2(trim_labels_file2, n, config.classes)
    a2 = np.sum(labels2, axis=0)
    # print(len(data1)/n)
    print(len(data2))
    print(a2)
    t2 = time.time()
    print(t2 - t1)
    # print(classes2)
    # for row, l in zip(data1, labels1):
    #     print(row[0], row[-1], l)

    # for matrix, l in zip(data2, labels2):
    #     for row in matrix:
    #         print(row[0], row[-1])
    #     print(matrix.shape)
    #     print(matrix[0, 0], matrix[38, 38])
    #     print(l)
    #     print()


def dump():
    t1 = time.time()
    train_data_f = 'data_1234_n'
    train_labels_f = 'labels_1234_n'
    test_data_f = 'data_5_n'
    test_labels_f = 'labels_5_n'
    trim_labels_file1 = 'trim_labels_1234.txt'
    n = 40
    data1, labels1 = load_data.load_data2d_1hot2(trim_labels_file1, n, config.classes)
    print(data1.shape, labels1.shape)
    trim_labels_file2 = 'trim_labels_5.txt'
    data2, labels2 = load_data.load_data2d_1hot2(trim_labels_file2, n, config.classes)
    print(data2.shape, labels2.shape)
    np.save(train_data_f + str(n), data1)
    np.save(train_labels_f + str(n), labels1)
    np.save(test_data_f + str(n), data2)
    np.save(test_labels_f + str(n), labels2)
    t2 = time.time()
    print(t2 - t1)


if __name__ == '__main__':
    get_test_data_with_sentences()
    # get_train_develop_train_data()
    # test_norm_and_div()
    # test_load_data2()
    # dump()
    # test_load_data2d_1hot()
    # test_load_data_1hot()
    # test_get_mfcc_path()
    # test_load_data()
    # test_get_classes_idx()
