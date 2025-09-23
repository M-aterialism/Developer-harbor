"""
RBM QUBO Quantum Image Denoising

这个文件是本项目的主函数。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import argparse
import numpy as np
from time import time

import params
import data_pretreat
import rbm_train
import noise_process
import qubo_inference
import evaluate

evaluate_features_path = params.evaluate_features_path
evaluate_labels_path = params.evaluate_labels_path
noisy_path = params.noisy_path
denoised_path = params.denoised_path
sigmas = params.sigmas
betas = params.betas
rhos = params.rhos


def main():
    """
    主函数入口

    Args:
    
    Returns:
        None
    """
    global evaluate_features_path, evaluate_labels_path, noisy_path, denoised_path, sigmas, betas, rhos
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='default')
    parser.add_argument('--rho_mode', default='sigma')
    ps = parser.parse_args()

    # 数据预处理
    if ps.mode == 'pretreat':
        print('Pretreating Data...')
        data_pretreat.data_pretreat()
        print('Completing!')

    # 训练RBM
    if ps.mode == 'train':
        print('Training RBM...')
        start_time = time()
        rbm_train.rbm_train()
        runtime = time() - start_time
        print(f'Completing! Total runtime is {runtime}.')

    # 数据加工
    if ps.mode == 'process':
        print('Adding Noise...')
        noisy_features_list, noisy_labels = noise_process.noise_process(sigmas)
        np.save(noisy_path + "noisy_features.npy", noisy_features_list)
        np.save(noisy_path + "noisy_labels.npy", noisy_labels)
        print('Completing!')

    # 图像去噪
    if ps.mode == 'inference':
        print('Removing Noise...')
        noisy_features_list = np.load(noisy_path + "noisy_features.npy")
        noisy_labels = np.load(noisy_path + "noisy_labels.npy")
        if ps.rho_mode == 'rho':
            denoised_features_list, denoised_labels = qubo_inference.denoising(noisy_features_list, noisy_labels, rhos)
            np.save(denoised_path + "denoised_features.npy", denoised_features_list)
            np.save(denoised_path + "denoised_labels.npy", denoised_labels)

        elif ps.rho_mode == 'sigma':
            for beta in betas:
                rhos = np.log((1 - beta * sigmas) / (beta * sigmas))
                denoised_features_list, denoised_labels = qubo_inference.denoising(noisy_features_list, noisy_labels, rhos)
                np.save(denoised_path + "denoised_features_" + str(round(beta, 3)) + ".npy", denoised_features_list)
                np.save(denoised_path + "denoised_labels_" + str(round(beta, 3)) + ".npy", denoised_labels)

        print('Completing!')

    # 数据后处理
    if ps.mode == 'evaluate':
        print('Evaluating...')
        noisy_features_list = np.load(noisy_path + "noisy_features.npy")
        noisy_labels = np.load(noisy_path + "noisy_labels.npy")
        evaluate_features_list = np.load(evaluate_features_path)
        evaluate_labels = np.load(evaluate_labels_path)
        if ps.rho_mode == 'rho':
            evaluate.evaluate(noisy_features_list, evaluate_features_list, sigmas, rhos)

        elif ps.rho_mode == 'sigma':
            for beta in betas:
                rhos = np.log((1 - beta * sigmas) / (beta * sigmas))
                evaluate.evaluate(noisy_features_list, evaluate_features_list, sigmas, rhos)

        print('Completing!')

    return


if __name__ == "__main__":
    main()
