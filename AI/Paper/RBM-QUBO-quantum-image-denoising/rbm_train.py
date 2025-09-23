"""
RBM QUBO Quantum Image Denoising

这个文件用于训练受限玻尔兹曼机。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np
from sklearn.neural_network._rbm import BernoulliRBM

import params

train_features_path = params.train_features_path
train_labels_path = params.train_labels_path
rbm_path = params.rbm_path
n_components = params.n_components
learning_rate = params.learning_rate
batch_size = params.batch_size
n_iter = params.n_iter
verbose = params.verbose
random_state = params.random_state


def rbm_train():
    """
    训练受限玻尔兹曼机

    Args:

    Returns:

    """
    global train_features_path, train_labels_path, rbm_path, n_components, \
        learning_rate, batch_size, n_iter, verbose, random_state

    # 获取训练数据
    train_features = np.load(train_features_path)
    train_labels = np.load(train_labels_path)

    # 数据列表长度
    num = len(train_labels)

    # 创建并配置BernoulliRBM模型
    rbm = BernoulliRBM(
        n_components=n_components,  # 隐藏单元数量
        learning_rate=learning_rate,  # 学习率
        batch_size=batch_size,  # 批量大小
        n_iter=n_iter,  # 迭代次数
        verbose=verbose,  # 显示训练进度
        random_state=random_state  # 随机种子
    )

    # 训练RBM模型
    print("Start training...")
    rbm.fit(train_features)
    print("Finish training!")

    # 将数据保存为二进制文件
    # print(rbm.components_.shape)  # (64, 144)
    # 保存RBM的参数时需要翻转符号（sklearn使用负值）
    np.save(rbm_path + "rbm_components_.npy", - rbm.components_)
    np.save(rbm_path + "rbm_intercept_hidden_.npy", - rbm.intercept_hidden_)
    np.save(rbm_path + "rbm_intercept_visible_.npy", - rbm.intercept_visible_)

    return

