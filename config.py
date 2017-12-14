data_path = '/home/ddy/projects/emotions/mfcc_data'

step_num0 = 10000

step_num1 = 100000

step_num2 = 200000

step_num3 = 400000

learning_rate0 = 0.05

learning_rate1 = 1e-2

learning_rate2 = 1e-3

learning_rate3 = 1e-4

label_file = './trim_labels_1.txt'

batch_size = 500

print_interval = 500

continue_eles = 50

train_rate = 0.7

classes_pickle = './pickles/classes.pickle'

gt_pickle = './pickles/gt.pickle'

pr_pickle = './pickles/pr.pickle'

# device = '/gpu:0'
visible_device = '2'

# input matrix n * 39
mfcc_n = 40

cnn_kernel1_shape = [5, 5, 1, 32]

cnn_kernel2_shape = [5, 5, 32, 64]

fc1_fs = 256

# n_classes = 6

classes = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

train_d_npy = './data_1234_n40.npy'

train_l_npy = './labels_1234_n40.npy'

test_d_npy = './data_5_n40.npy'

test_l_npy = './labels_5_n40.npy'
