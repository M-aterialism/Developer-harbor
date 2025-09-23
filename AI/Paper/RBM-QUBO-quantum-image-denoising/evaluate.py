"""
RBM QUBO Quantum Image Denoising

这个文件用于评估去噪方法的性能。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np

import params


def evaluate(noisy_features_list, denoised_features_list, sigmas, rhos):
    """
    训练受限玻尔兹曼机

    Args:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表

    Returns:

    """
    for i in range(len(noisy_features_list)):
        overlap = 1 - np.mean(np.abs(noisy_features_list[i, :] - denoised_features_list[i, :]))
        print(f"sigma: {sigmas[i]}, rho: {rhos[i]}, overlap: {overlap}.")

    return

