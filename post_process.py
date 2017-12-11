import pickle
import numpy as np


def dump_list(l, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(l, pickle_f)


def load_pickle(pickle_name):
    r = None
    with open(pickle_name, 'rb') as pickle_f:
        r = pickle.load(pickle_f)
    return r


def process_results(gt, pr, classes):
    LEN = len(classes)
    matrix = np.zeros((LEN, LEN))
    for i, j in zip(gt, pr):
        matrix[gt[i], pr[j]] += 1
    return matrix, classes
