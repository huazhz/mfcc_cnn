id_str = 'h01031737'

visible_device = '2'

is_train = True

is_restore = False

restore_file = ''

restart_epoch_i = 0

loop_epoch_nums = [50, 50, 50, 50]

learning_rates = [0.1, 0.05, 0.02, 0.01]

log_epoch_interval = 10

persist_checkpoint_interval = 50
persist_checkpoint_file = 'h-p-my-model/p-my-model' + id_str + '_'

# classes = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
classes = ['neu', 'ang', 'hap', 'sad']

loss_weights = [1, 1, 1, 1]

batch_size = 500

train_k_prob = 0.5

gt_pickle = './pickles/gt_' + id_str + '.pickle'

pr_pickle = './pickles/pr_' + id_str + '.pickle'

# input matrix n * 39
mfcc_n = 200

conv_strides = [1, 1, 1, 1]

maxpool_ksize = [1, 2, 2, 1]

maxpool_strides = [1, 2, 2, 1]

cnn_kernel1_shape = [5, 5, 1, 16]

cnn_kernel2_shape = [5, 5, 16, 32]

fc1_fs = 512


train_d_npy = './npy2/data_123_n200_s40.npy'

train_l_npy = './npy2/ls_1hot_123_n200_s40.npy'

vali_d_npy = './npy2/data_4_n200_s40.npy'

vali_l_npy = './npy2/ls_1hot_4_n200_s40.npy'

test_d_npy = './npy2/data_5_n200_s40.npy'

test_l_npy = './npy2/ls_1hot_5_n200_s40.npy'

# # old config below

print_interval = 1000

save_checkpoint_interval = 10000

step_num0 = 100000

step_num1 = 100000

step_num2 = 100000

step_num3 = 0

learning_rate0 = 0.05

learning_rate1 = 0.03

learning_rate2 = 0.01

learning_rate3 = 1e-4

checkpoint_file = './my-model2/my-model'

# data_path = '/home/dai/Projects/emotions/data'
#
#
#
# label_file = './trim_labels_1.txt'
#
#
#
# continue_eles = 50
#
# train_rate = 0.7
#
# classes_pickle = './pickles/classes.pickle'

