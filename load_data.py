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


