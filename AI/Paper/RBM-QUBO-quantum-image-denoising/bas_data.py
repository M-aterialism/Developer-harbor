"""
RBM QUBO Quantum Image Denoising

这个文件用于数据生成（BAS数据集）。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import numpy as np
from PIL import Image

import params

train_feature_path = params.train_feature_path
train_label_path = params.train_label_path
test_feature_path = params.test_feature_path
test_label_path = params.test_label_path
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


def gen_bars(pix):
    """
    生成一幅条形图像向量

    Args:
        pix: int, 图像的像素数

    Returns:
        image: array[array[int]], 图像的像素矩阵
    """
    m = round(np.sqrt(pix))
    image = np.zeros((m, m))
    for i in range(m):
        if np.random.randint(0, 2) == 1:
            image[:, i] = np.ones(m)

    return np.ndarray.flatten(image)


def gen_stripes(pix):
    """
    生成一幅条纹图像向量

    Args:
        pix: int, 图像的像素数

    Returns:
        image: array[array[int]], 图像的像素矩阵
    """
    m = round(np.sqrt(pix))
    image = np.zeros((m, m))
    for i in range(m):
        if np.random.randint(0, 2) == 1:
            image[i, :] = np.ones(m)

    return np.ndarray.flatten(image)


def data_generate():
    """
    生成条形或条纹图像数据集

    Args:

    Returns:

    """
    global train_feature_path, train_label_path, test_feature_path, test_label_path, result_list_path, number_select

    # 设定随机种子、图像的像素数和数据集的总容量
    np.random.seed(1)
    pix = 12 ** 2
    n = 4000
    features = np.empty((n, pix), dtype=int)
    labels = np.empty(n, dtype=int)

    # 生成数据标签
    for i in range(n):
        labels[i] = np.random.randint(0, 2)

    # 生成数据图像
    for i in range(n):
        if labels[i] == 0:
            features[i, :] = np.copy(gen_bars(pix))
        else:
            features[i, :] = np.copy(gen_stripes(pix))

    # 按比例分割出训练集和测试集，并保存
    split = round(0.75 * n)

    train_feats = features[0:split, :]
    train_labels = labels[0:split]

    test_feats = features[split:, :]
    test_labels = labels[split:]

    np.save(train_feature_path, train_feats)
    np.save(train_label_path, train_labels)

    for i in range(len(number_select)):
        path = result_list_path + "true_bas_" + str(number_select[i]) + "_" \
               + str(test_labels[number_select[i]]) + ".png"
        display(test_feats[number_select[i]], path)

    # np.save(test_feature_path, test_feats)
    # np.save(test_label_path, test_labels)

    return test_feats, test_labels
