"""
RBM QUBO Quantum Image Denoising

这个文件用于处理噪声数据。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np
from PIL import Image

import params

test_features_path = params.test_features_path
test_labels_path = params.test_labels_path
result_list_path = params.result_list_path
number_select = params.number_select


def display(feature, path):
    """
    展示像素向量对应的图像

    Args:
        feature: array[int], 像素向量
        path: string, 展示路径

    Returns:

    """
    pixroot = int(np.round(np.sqrt(len(feature))))
    feature = 255 * np.reshape(feature, (pixroot, pixroot))
    # 画一幅图像
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(feature.astype(np.uint8), params.file_mode)  # 'L' 表示灰度模式
    # 显示图像
    # image.show()
    # 保存图像（可选）为 JPG 文件
    image.save(path)

    return


def add_salt_and_pepper(features, sigma):
    """
    给像素向量添加椒盐噪声

    Args:
        features: array[array[int]], 像素向量列表
        sigma: float, 噪声参数

    Returns:
        features: array[array[int]], 处理后的像素向量列表
    """
    # 添加椒盐噪声
    for i in range(len(features)):
        swaps = np.random.uniform(low=0, high=1, size=features[i].shape)
        mask = (swaps < sigma)
        features[i] = (features[i] + mask) % 2

    return features


def noise_process(sigmas):
    """
    噪声过程

    Args:
        sigmas: array[float], 噪声参数列表

    Returns:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
    """
    global test_features_path, test_labels_path, result_list_path, number_select

    # 获取测试数据
    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)

    # 制作真实数据
    true_features = test_features[number_select]
    true_labels = test_labels[number_select]
    for i in range(len(number_select)):
        path = result_list_path + "true_" + str(number_select[i]) + "_" + str(true_labels[i]) + ".png"
        display(true_features[i], path)

    # 制作噪声数据
    noisy_features_list = []
    noisy_labels = true_labels
    for sigma in sigmas:
        noisy_features_list.append(add_salt_and_pepper(true_features, sigma))
        for i in range(len(number_select)):
            path = result_list_path + "noisy_" + str(number_select[i]) + "_" + str(noisy_labels[i]) \
                   + "_" + str(round(sigma, 3)) + ".png"
            display(noisy_features_list[-1][i], path)

    noisy_features_list = np.array(noisy_features_list)
    return noisy_features_list, noisy_labels

