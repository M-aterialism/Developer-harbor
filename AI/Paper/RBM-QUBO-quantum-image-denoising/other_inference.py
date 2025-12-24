"""
RBM QUBO Quantum Image Denoising

这个文件用于其它对比方法推理去噪图像。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import igraph
import numpy as np
from PIL import Image
from skimage import filters
from sklearn.neural_network._rbm import BernoulliRBM

import params

rbm_path = params.rbm_path
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


def median_inference(noisy_features_list, noisy_labels, sigmas):
    """
    中值滤波推理去噪图像

    Args:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
        noisy_labels: array[int], 带噪声的像素标签列表
        sigmas: array[float], 去噪参数列表

    Returns:
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表
        denoised_labels: array[int], 去噪像素标签列表
    """
    global result_list_path, number_select

    # 制作去噪数据
    denoised_features_list = []
    denoised_labels = noisy_labels
    for i in range(len(noisy_features_list)):
        noisy_features = noisy_features_list[i]
        sigma = sigmas[i]
        denoised_features = []
        denoised_median = np.empty(noisy_features[0].shape)
        pixroot = int(np.round(np.sqrt(len(noisy_features[0]))))
        for j in range(len(number_select)):
            img_noisy = np.reshape(noisy_features[number_select[j]], (pixroot, pixroot))
            fixedimg = filters.median(img_noisy)
            denoised_median = np.reshape(fixedimg, denoised_median.shape)
            path = result_list_path + "denoised_median_" + str(number_select[j]) + "_" + \
                str(denoised_labels[number_select[j]]) + "_" + str(round(sigma, 3)) + ".png"
            display(denoised_median, path)
            denoised_features.append(denoised_median)

        denoised_features_list.append(denoised_features)

    denoised_features_list = np.array(denoised_features_list)
    return denoised_features_list, denoised_labels


def gaussian_inference(noisy_features_list, noisy_labels, sigmas):
    """
    高斯滤波推理去噪图像

    Args:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
        noisy_labels: array[int], 带噪声的像素标签列表
        sigmas: array[float], 去噪参数列表

    Returns:
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表
        denoised_labels: array[int], 去噪像素标签列表
    """
    global result_list_path, number_select

    # 制作去噪数据
    denoised_features_list = []
    denoised_labels = noisy_labels
    for i in range(len(noisy_features_list)):
        noisy_features = noisy_features_list[i]
        sigma = sigmas[i]
        denoised_features = []
        denoised_gaussian = np.empty(noisy_features[0].shape)
        pixroot = int(np.round(np.sqrt(len(noisy_features[0]))))
        for j in range(len(number_select)):
            img_noisy = np.reshape(noisy_features[number_select[j]], (pixroot, pixroot))
            fixedimg = np.round(filters.gaussian(img_noisy))
            denoised_gaussian = (np.reshape(fixedimg, denoised_gaussian.shape))
            path = result_list_path + "denoised_gaussian_" + str(number_select[j]) + "_" + \
                str(denoised_labels[number_select[j]]) + "_" + str(round(sigma, 3)) + ".png"
            display(denoised_gaussian, path)
            denoised_features.append(denoised_gaussian)

        denoised_features_list.append(denoised_features)

    denoised_features_list = np.array(denoised_features_list)
    return denoised_features_list, denoised_labels


def create_graph_for_flow(img, k=1, lam=3):
    """
    创建最小最大去噪图

    Args:
        img: array[array[int]], 带噪声的12×12图像
        k: float, 边初始权重
        lam: float, 边初始权重

    Returns:
        edge_list: array[tuple], 边列表
        weights: array[float], 边权重列表
        s: int, 源最大数量
        t: int, 目标最大数量
    """
    max_num = len(img) * len(img[0])
    s, t = max_num, max_num + 1
    edge_list = []
    weights = []
    for r_idx, row in enumerate(img):
        for idx, pixel in enumerate(row):
            px_id = (r_idx * len(row)) + idx
            # add edge to cell to the left
            if px_id != 0:
                edge_list.append((px_id - 1, px_id))
                weights.append(k)

            # add edge to cell to the right
            if px_id != len(row) - 1:
                edge_list.append((px_id + 1, px_id))
                weights.append(k)

            # add edge to cell to the above
            if r_idx != 0:
                edge_list.append((px_id - len(row), px_id))
                weights.append(k)

            # add edge to cell to the below
            if r_idx != len(img) - 1:
                edge_list.append((px_id + len(row), px_id))
                weights.append(k)

            # add an edge to either s (source) or t (sink)
            if pixel == 1:
                edge_list.append((s, px_id))
                weights.append(lam)

            else:
                edge_list.append((px_id, t))
                weights.append(lam)

    return edge_list, weights, s, t


def flow_recover(noisy, k=1, lam=3.5):
    """
    去噪工具，最小最大流恢复

    Args:
        noisy: array[array[int]], 带噪声的12×12图像
        k: float, 边初始权重
        lam: float, 边初始权重

    Returns:
        recovered: array[array[array[int]]], 去噪的12×12图像
    """
    edge_list, weights, s, t = create_graph_for_flow(noisy, k, lam)
    g = igraph.Graph(edge_list)
    output = g.maxflow(s, t, weights)
    recovered = np.array(output.membership[:-2]).reshape(noisy.shape)
    # flip because of implementation 0-1
    recovered = np.mod(recovered + 1, 2)
    return recovered


def flow_inference(noisy_features_list, noisy_labels, sigmas):
    """
    图割方法推理去噪图像

    Args:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
        noisy_labels: array[int], 带噪声的像素标签列表
        sigmas: array[float], 去噪参数列表

    Returns:
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表
        denoised_labels: array[int], 去噪像素标签列表
    """
    global result_list_path, number_select

    # 制作去噪数据
    denoised_features_list = []
    denoised_labels = noisy_labels
    for i in range(len(noisy_features_list)):
        noisy_features = noisy_features_list[i]
        sigma = sigmas[i]
        denoised_features = []
        denoised_flow = np.empty(noisy_features[0].shape)
        pixroot = int(np.round(np.sqrt(len(noisy_features[0]))))
        for j in range(len(number_select)):
            img_noisy = np.reshape(noisy_features[number_select[j]], (pixroot, pixroot))
            # can give error if img is pure white (rare)
            fixedimg = flow_recover(img_noisy)
            denoised_flow = (np.reshape(fixedimg, denoised_flow.shape))
            path = result_list_path + "denoised_flow_" + str(number_select[j]) + "_" + \
                str(denoised_labels[number_select[j]]) + "_" + str(round(sigma, 3)) + ".png"
            display(denoised_flow, path)
            denoised_features.append(denoised_flow)

        denoised_features_list.append(denoised_features)

    denoised_features_list = np.array(denoised_features_list)
    return denoised_features_list, denoised_labels


def gibbs_inference(noisy_features_list, noisy_labels, sigmas):
    """
    Gibbs方法推理去噪图像

    Args:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
        noisy_labels: array[int], 带噪声的像素标签列表
        sigmas: array[float], 去噪参数列表

    Returns:
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表
        denoised_labels: array[int], 去噪像素标签列表
    """
    global rbm_path, result_list_path, number_select

    # 获取RBM参数
    rbm_components = np.load(rbm_path + "rbm_components_.npy")
    rbm_intercept_hidden = np.load(rbm_path + "rbm_intercept_hidden_.npy")
    rbm_intercept_visible = np.load(rbm_path + "rbm_intercept_visible_.npy")

    # 创建并配置BernoulliRBM模型
    rbm = BernoulliRBM(random_state=0, verbose=True)
    [rbm.components_, rbm.intercept_hidden_, rbm.intercept_visible_] = \
        [-1 * rbm_components, -1 * rbm_intercept_hidden, -1 * rbm_intercept_visible]

    # 制作去噪数据
    denoised_features_list = []
    denoised_labels = noisy_labels
    for i in range(len(noisy_features_list)):
        noisy_features = noisy_features_list[i]
        sigma = sigmas[i]
        denoised_features = []
        gibbs_steps = 20  # Number of gibbs steps
        alpha = 0.8  # decay factor for averaging
        denoised_gibbs = np.empty(noisy_features[0].shape)
        pixroot = int(np.round(np.sqrt(len(noisy_features[0]))))
        for j in range(len(number_select)):
            b = rbm.gibbs(noisy_features[number_select[j]])
            x_gibbs = np.zeros(pixroot ** 2) + np.copy(b)
            for k in range(gibbs_steps):
                b = rbm.gibbs(b)
                # Averaging the images
                x_gibbs += (alpha ** (k + 1)) * b.astype(float)

            # create the final image based on threshold
            denoised_gibbs = np.where(x_gibbs > 0.5 * np.max(x_gibbs), 1, 0)
            path = result_list_path + "denoised_gibbs_" + str(number_select[j]) + "_" + \
                str(denoised_labels[number_select[j]]) + "_" + str(round(sigma, 3)) + ".png"
            display(denoised_gibbs, path)
            denoised_features.append(denoised_gibbs)

        denoised_features_list.append(denoised_features)

    denoised_features_list = np.array(denoised_features_list)
    return denoised_features_list, denoised_labels
