import struct
import config
import os
import HTK
from collections import Counter

emofolds = ['session1_emoevaluation', 'session2_emoevaluation', 'session3_emoevaluation', 'session4_emoevaluation', ]
emof_suffix = '.txt'

emos = ['neu', 'ang', 'hap', 'sad']


def process_emof_with_filter(emo_file, emo_list):
    with open(emo_file, 'r') as emo_f:
        for line in emo_f:
            if '[' in line and ']' in line and 'Ses' in line:
                for emo in emo_list:
                    # print(line)
                    if emo in line:
                        eles = line.split()
                        # print(len(eles))
                        print(eles[3], eles[4])
                        break


def get_filter_labels(a_fs, emos):
    for a_f in a_fs:
        process_emof_with_filter(a_f, emos)


def process_emof(emo_filename):
    with open(emo_filename, 'r') as emo_f:
        i = 0
        for line in emo_f:
            if '[' in line and ']' in line and 'Ses' in line:
                eles = line.split()
                # print(eles[3], eles[4])
                if eles[4] != 'xxx':
                    print(eles[3], eles[4])
            i += 1


def process_emof_trim(emo_filename):
    with open(emo_filename, 'r') as emo_f:
        i = 0
        for line in emo_f:
            if '[' in line and ']' in line and 'Ses' in line:
                eles = line.split()
                # print(eles[3], eles[4])
                if eles[4] != 'xxx' and eles[4] != 'dis' and eles[4] != 'oth' and eles[4] != 'fea' and eles[4] != 'sur':
                    print(eles[3], eles[4])
            i += 1


# every element in folds is a relative path
def list_txtf(folds):
    fs = list()
    for fold_name in folds:
        # fold is the absolute path
        fold = os.path.join(config.data_path, fold_name)
        fs += [os.path.join(fold, f) for f in os.listdir(fold) if emof_suffix in f]
    return fs


# every element in a_fs is absolute file
# save the result use standard output redirect, python process_rawdata.py > simple_labels.txt
def get_emo_labels(a_fs):
    for a_f in a_fs:
        process_emof(a_f)


def get_trim_emo_labels(a_fs):
    for a_f in a_fs:
        process_emof_trim(a_f)


# {'sad', 'oth', 'hap', 'neu', 'exc', 'ang', 'fea', 'dis', 'sur', 'fru'}
def count_labels(sim_label_f):
    labels = list()
    with open(sim_label_f, 'r') as f:
        for line in f:
            eles = line.split()
            if len(eles) == 2:
                labels.append(eles[1])
    return Counter(labels)


def open_mfcc_f(mfcc_f):
    htk = HTK.HTKFile()
    htk.load(mfcc_f)
    return htk.data


if __name__ == '__main__':
    # emo_filename_ = os.path.join(config.data_path, 'session1_emoevaluation/Ses01F_impro01.txt')
    # print(emo_filename_)
    # process_emof(emo_filename_)
    # results = list_txtf(emofolds)
    # print(results)

    # run the following with: python process_rawdata.py > simple_labels.txt
    fs_ = list_txtf(emofolds)
    get_filter_labels(fs_, emos)
    # get_trim_emo_labels(fs_)

    # simple_label_f = './simple_labels_all.txt'
    # cs = count_labels(simple_label_f)
    # print(len(cs))
    # print(cs)
    #
    # simple_label_f = './trim_labels_2.txt'
    # cs = count_labels(simple_label_f)
    # print(len(cs))
    # print(cs)
    # htk_f = os.path.join(config.data_path, 'mfcc_iemocap/Ses03F_script03_2/Ses03F_script03_2_F001.mfcc')
    # data = open_mfcc_f(htk_f)
    # # print(len(data))
    # for row in data:
    #     print(len(row))
    # pass
