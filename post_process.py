import pickle


def dump_list(l, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(l, pickle_f)


def load_pickle(pickle_name):
    r = None
    with open(pickle_name, 'rb') as pickle_f:
        r = pickle.load(pickle_f)
    return r
