import load_data


def test_get_mfcc_path():
    mfcc_key = 'Ses01M_impro06_F002'
    result = load_data.get_mfcc_path(mfcc_key)
    with open(result, 'rb') as f:
        print(f.read())


def test_load_data():
    simple_labels_file = 'test_labels.txt'
    data, labels = load_data.load_data(simple_labels_file)
    print(data.shape)
    print(labels.shape)
    # for row in data:
    #     print(row)
    # print(type(labels))
    # for label in labels:
    #     print(type(label))


if __name__ == '__main__':
    # test_get_mfcc_path()
    test_load_data()