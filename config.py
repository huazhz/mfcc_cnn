id_str = '01231059'

is_log_cfg = True

visible_device = '3'

is_train = False

is_restore = False

restore_file = 'p-my-model/p-my-model01231059_600'

restart_epoch_i = 0

loop_epoch_nums = [100, 200, 200, 300, 300]

learning_rates = [0.2, 0.1, 0.05, 0.02, 0.01]

log_epoch_interval = 5

persist_checkpoint_interval = 300
persist_checkpoint_file = 'p-my-model/p-my-model' + id_str + '_'

# classes = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
classes = ['neu', 'ang', 'hap', 'sad']

loss_weights = [0.68, 0.95, 1.92, 1.03]

batch_size = 512

train_k_prob = 0.2

gt_pickle = './pickles/gt_' + id_str + '.pickle'

pr_pickle = './pickles/pr_' + id_str + '.pickle'

gt_sens = 'gt_sens_' + id_str + '.npy'
pr_sens = 'pr_sens_' + id_str + '.npy'

# input matrix n * 39
mfcc_n = 200

conv_strides = [1, 1, 1, 1]

maxpool_ksize = [1, 4, 2, 1]

maxpool_strides = [1, 4, 2, 1]

cnn_kernel1_shape = [10, 5, 1, 16]

cnn_kernel2_shape = [10, 5, 16, 32]

fc1_fs = 256


train_d_npy = './npy2/data_123_n200_s40.npy'

train_l_npy = './npy2/ls_1hot_123_n200_s40.npy'

vali_d_npy = './npy2/data_4_n200_s40.npy'

vali_l_npy = './npy2/ls_1hot_4_n200_s40.npy'

test_d_npy = './npy2/data_5_sens_n200_s40.npy'

test_l_npy = './npy2/ls_1hot_5_sens_n200_s40.npy'

test_sens_npy = './npy2/sens_5_sens_n200_s40.npy'

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

data_path = '/home/dai/Projects/emotions/data'

label_file = './trim_labels_1.txt'

continue_eles = 50

train_rate = 0.7

classes_pickle = './pickles/classes.pickle'

