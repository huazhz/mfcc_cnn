import post_process


def test_load():
    classes = post_process.load_pickle('./pickles/classes.pickle')
    gt = post_process.load_pickle('./pickles/gt.pickle')
    pr = post_process.load_pickle('./pickles/pr.pickle')
    print(classes)
    print(gt)
    print(pr)


def test_process_results():
    classes = post_process.load_pickle('./pickles/classes.pickle')
    gt = post_process.load_pickle('./pickles/gt.pickle')
    pr = post_process.load_pickle('./pickles/pr.pickle')
    matrix, classes = post_process.process_results(gt, pr, classes)
    print(classes)
    # print(matrix)
    for row in matrix:
        # print(row)
        strs = [str(i) for i in row]
        print_alist(strs)


def print_alist(a):
    for i in a:
        print(i, end='\t')
    print()


if __name__ == '__main__':
    # test_load()
    test_process_results()
