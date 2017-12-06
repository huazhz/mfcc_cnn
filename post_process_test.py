import post_process


def test_load():
    classes = post_process.load_pickle('./pickles/classes.pickle')
    gt = post_process.load_pickle('./pickles/gt.pickle')
    pr = post_process.load_pickle('./pickles/pr.pickle')
    print(classes)
    print(gt)
    print(pr)


if __name__ == '__main__':
    test_load()
