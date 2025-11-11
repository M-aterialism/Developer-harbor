"""
RBM QUBO Quantum Image Denoising

这个文件用于评估QUBO去噪方法的性能。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np

import params

number_select = params.number_select


def evaluate(true_features_list, denoised_features_list, sigmas, rhos):
    """
    评估去噪图像和真实图像的重叠度

    Args:
        true_features_list: array[array[array[int]]], 带噪声的像素向量列表
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表
        sigmas: array[float], 去噪参数列表
        rhos: array[float], 去噪参数列表

    Returns:

    """
    global number_select

    for i in range(len(denoised_features_list)):
        overlap = 1 - np.mean(np.abs(true_features_list[number_select] - denoised_features_list[i, :]))
        print(f"sigma: {sigmas[i]}, rho: {rhos[i]}, overlap: {overlap}.")

    return
