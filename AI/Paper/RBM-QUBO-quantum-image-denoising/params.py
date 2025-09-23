"""
RBM QUBO Quantum Image Denoising

这个文件用于保存参数。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np

# 文件数据参数
train_list_path = 'mnist_jpg/train/'
test_list_path = 'mnist_jpg/test/'

train_features_path = 'data/MNIST_train_features.npy'
train_labels_path = 'data/MNIST_train_labels.npy'
test_features_path = 'data/MNIST_test_features.npy'
test_labels_path = 'data/MNIST_test_labels.npy'
evaluate_features_path = 'data/denoised_features_0.75.npy'
evaluate_labels_path = 'data/denoised_labels_0.75.npy'

noisy_path = 'data/'
denoised_path = 'data/'
rbm_path = 'data/'
result_list_path = 'result/'
file_mode = 'L'  # 'L' 表示灰度模式，'RGB' 表示彩色模式

# 训练参数
n_components = 64  # 隐藏单元数量
learning_rate = 0.01  # 学习率
batch_size = 50  # 批量大小
n_iter = 150  # 迭代次数
verbose = True  # 显示训练进度
random_state = 42  # 随机种子

# 推理参数
sigmas = np.array([0.2])  # np.linspace(0.025, 0.25, num=10)
betas = [0.75]  # sigma模式下设置
rhos = np.log((1 - sigmas) / sigmas)  # rho模式下设置，需和sigmas大小匹配
number_select = np.linspace(0, 9, num=10).astype(int)

# 模拟退火求解器参数
anneal_initial_temperature = 100  # 初始温度
anneal_alpha = 0.95  # 降温系数
anneal_cutoff_temperature = 0.01  # 截止温度
anneal_iterations_per_t = 10000  # 每个温度迭代深度
anneal_size_limit = 10  # 输出解的个数，默认输出100个解
anneal_verbose = False  # 是否在控制台输出计算进度，默认False
anneal_rand_seed = 42  # numpy随机数生成器的随机种子
anneal_process_num = 1  # 并行进程数 (-1为自动调用所有可用核心，1为单进程)，默认为1

# CIM真机求解器参数
save_dir_path = 'tmp/'
machine_name = 'CPQC-550'
cim_bit = 550
