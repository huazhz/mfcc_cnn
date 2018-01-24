import numpy as np
import config
import sys


def process_result(gt, pr, classes):
    LEN = len(classes)
    matrix = np.zeros((LEN, LEN))
    for i, j in zip(gt, pr):
        matrix[i, j] += 1
    return matrix, classes


# def print_csv():
#     gt = np.load(config.gt_sens)
#     pr = np.load(config.pr_sens)
#     total_acc = np.sum(gt == pr) / len(gt)
#     matrix, classes = process_result(gt, pr, config.classes)
#     print()
#     print('  a\\p', end='\t')
#     for c in classes:
#         print(c, end='\t')
#     print()
#     for i in range(len(classes)):
#         print(' ', classes[i], end='\t')
#         for ele in matrix[i]:
#             print(ele, end='\t')
#         print()
#     print()
#
#     sum_1 = np.sum(matrix, axis=1)
#     matrix2 = matrix / sum_1.reshape((-1, 1))
#
#     print('  a\\p', end='\t')
#     for c in classes:
#         print(' ', c, end='\t')
#     print()
#     for i in range(len(classes)):
#         print(' ', classes[i], end='\t')
#         for ele in matrix2[i]:
#             print('%.4f' % ele, end='\t')
#         print()
#     print()
#
#     avg = 0
#     for i in range(len(classes)):
#         avg += matrix2[i, i]
#     print('  average accurate is %.4f' % (avg / len(classes)))
#     print('  total accurate is %.4f' % float(total_acc))
#     print()


def print_csv_with_id(id_str):
    if id_str is None:
        id_str = config.id_str
    print('  id_str', id_str)
    gt_sens = 'gt_sens_' + id_str + '.npy'
    pr_sens = 'pr_sens_' + id_str + '.npy'
    gt = np.load(gt_sens)
    pr = np.load(pr_sens)
    total_acc = np.sum(gt == pr) / len(gt)
    matrix, classes = process_result(gt, pr, config.classes)
    print()
    print('  a\\p', end='\t')
    for c in classes:
        print(c, end='\t')
    print()
    for i in range(len(classes)):
        print(' ', classes[i], end='\t')
        for ele in matrix[i]:
            print(ele, end='\t')
        print()
    print()

    sum_1 = np.sum(matrix, axis=1)
    matrix2 = matrix / sum_1.reshape((-1, 1))

    print('  a\\p', end='\t')
    for c in classes:
        print(' ', c, end='\t')
    print()
    for i in range(len(classes)):
        print(' ', classes[i], end='\t')
        for ele in matrix2[i]:
            print('%.4f' % ele, end='\t')
        print()
    print()

    avg = 0
    for i in range(len(classes)):
        avg += matrix2[i, i]
    print('  average accurate is %.4f' % (avg / len(classes)))
    print('  total accurate is %.4f' % float(total_acc))
    print()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print_csv_with_id(sys.argv[1])
    else:
        print_csv_with_id(None)