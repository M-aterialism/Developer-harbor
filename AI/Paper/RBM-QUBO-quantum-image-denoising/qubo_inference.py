"""
RBM QUBO Quantum Image Denoising

这个文件用于求解QUBO问题以推理去噪图像。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np
from PIL import Image

import kaiwu as kw

import params

rbm_path = params.rbm_path
result_list_path = params.result_list_path
number_select = params.number_select

# 授权初始化代码
# 示例的user_id和sdk_code无效，需要替换成自己的用户ID和SDK授权码
kw.license.init(user_id="77502211722072066", sdk_code="yeZw5s4c5cs4pbv6jNb91JH0koJosB")


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


def construct_tilde_qubo(components, intercept_hidden, intercept_visible, noisy_feature, rho):
    """
    构造QUBO系数矩阵

    Args:
        components: array[array[float]], RBM权重
        intercept_hidden: array[float], RBM隐藏层偏置
        intercept_visible: array[float], RBM可视层偏置
        noisy_feature: array[float], 带噪声的像素向量
        rho: float, 去噪参数

    Returns:
        qubo: array[array[float]], QUBO系数矩阵
    """
    # 构建QUBO矩阵，注意满足上三角矩阵的条件
    nh = len(intercept_hidden)
    nv = len(intercept_visible)
    n = nh + nv
    qubo = np.zeros((n, n))
    intercept_visible = intercept_visible + rho * (1 - 2 * noisy_feature)
    for i in range(nv):
        for j in range(nv, n):
            qubo[i][j] = components[j-nv][i]
            qubo[j][j] = intercept_hidden[j-nv]

        qubo[i][i] = intercept_visible[i]

    return qubo


def solve_qubo_anneal(qubo):
    """
    调用kaiwu经典求解器求解QUBO问题

    Args:
        qubo: array[array[float]], QUBO系数矩阵

    Returns:
        res: array[int], 最优0-1向量
    """
    ising = kw.qubo.qubo_matrix_to_ising_matrix(qubo)
    solver = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=params.anneal_initial_temperature,
                                                      alpha=params.anneal_alpha,
                                                      cutoff_temperature=params.anneal_cutoff_temperature,
                                                      iterations_per_t=params.anneal_iterations_per_t,
                                                      size_limit=params.anneal_size_limit)
    solution = solver.solve(ising[0])
    if solution.shape == (0,):
        print("求解器未找到解！")
        return solution

    opt = kw.sampler.optimal_sampler(ising[0], solution, 0)

    if opt[0][0][-1] == 1:
        res = opt[0][0][:-1]
        res = (res + 1) / 2
    else:
        res = opt[0][0][:-1]
        res = (-res + 1) / 2

    return res


def solve_qubo_cim(qubo):
    """
    调用kaiwu真机CIM求解器求解QUBO问题

    Args:
        qubo: array[array[float]], QUBO系数矩阵

    Returns:
        res: array[int], 最优0-1向量
    """
    # 根据比特数确定机器名
    cim_bit = params.cim_bit
    machine_name = params.machine_name
    save_dir_path = params.save_dir_path
    assert ((cim_bit == 100 and machine_name == "CPQC-100")
            or (cim_bit == 550 and machine_name == "CPQC-550")), "CIM真机名称错误"

    kw.common.CheckpointManager.save_dir = save_dir_path

    # 调整矩阵精度，否则精度校验会出现“CSV数据文件的精度过高”问题
    qubo_precision = kw.qubo.adjust_qubo_matrix_precision(qubo)

    ising = kw.qubo.qubo_matrix_to_ising_matrix(qubo_precision)
    optimizer = kw.cim.CIMOptimizer(user_id='1241241515', sdk_code='absd1232',
                                    task_name="test", machine_name=params.machine_name)
    solution = optimizer.solve(ising[0])
    opt = kw.sampler.optimal_sampler(ising[0], solution, 0)

    if opt[0][0][-1] == 1:
        res = opt[0][0][:-1]
        res = (res + 1) / 2
    else:
        res = opt[0][0][:-1]
        res = (-res + 1) / 2

    return res


def qubo_inference(noisy_features_list, noisy_labels, rhos):
    """
    QUBO方法推理去噪图像

    Args:
        noisy_features_list: array[array[array[int]]], 带噪声的像素向量列表
        noisy_labels: array[int], 带噪声的像素标签列表
        rhos: array[float], 去噪参数列表

    Returns:
        denoised_features_list: array[array[array[int]]], 去噪像素向量列表
        denoised_labels: array[int], 去噪像素标签列表
    """
    global rbm_path, result_list_path, number_select

    # 获取RBM参数
    rbm_components = np.load(rbm_path + "rbm_components_.npy")
    rbm_intercept_hidden = np.load(rbm_path + "rbm_intercept_hidden_.npy")
    rbm_intercept_visible = np.load(rbm_path + "rbm_intercept_visible_.npy")

    # 制作去噪数据
    denoised_features_list = []
    denoised_labels = noisy_labels
    for i in range(len(noisy_features_list)):
        noisy_features = noisy_features_list[i]
        rho = rhos[i]
        denoised_features = []
        for j in range(len(number_select)):
            # 构造QUBO系数矩阵
            qubo = construct_tilde_qubo(rbm_components, rbm_intercept_hidden, rbm_intercept_visible,
                                        noisy_features[number_select[j]], rho)
            # 求解器求解QUBO问题
            vh = solve_qubo_anneal(qubo)
            denoised_feature = vh[:len(noisy_features[number_select[j]])]
            path = result_list_path + "denoised_" + str(number_select[j]) + "_" + \
                str(denoised_labels[number_select[j]]) + "_" + str(round(rho, 3)) + ".png"
            display(denoised_feature, path)
            denoised_features.append(denoised_feature)

        denoised_features_list.append(denoised_features)

    denoised_features_list = np.array(denoised_features_list)
    return denoised_features_list, denoised_labels
