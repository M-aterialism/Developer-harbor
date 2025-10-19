"""
RBM QUBO Quantum Image Denoising

这个文件用于数据预处理。

作者：周澍锦
联系方式：Your_Beatitude@189.cn
"""

import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import minmax_scale

import params

train_list_path = params.train_list_path
test_list_path = params.test_list_path
train_feature_path = params.train_feature_path
train_label_path = params.train_label_path
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


def label_get(file_name):
    """
    读取文件标签

    Args:
        file_name: string, 文件名称

    Returns:
        serial_number: int, 文件序号
        label: int, 文件标签
    """
    name_without_ext = file_name.split('.')[0]
    parts = name_without_ext.split('_')

    serial_number = int(parts[1])  # 序号
    label = int(parts[2])  # 标签

    return serial_number, label


def image_read(image_path):
    """
    读取图像文件

    Args:
        image_path: string, 图像路径

    Returns:
        pixels: array[array[int]], 图像数据
    """
    # 根据文件编码方式打开
    image = Image.open(image_path)
    pixels = np.array(image)

    return pixels


def create_12x12_binary(pixels_list, label_list):
    """
    创建12×12二值化数据集

    Args:
        pixels_list: array[array[array[int]]], 处理前的28×28图像列表
        label_list: array[int], 图像标签列表

    Returns:
        feats_12x12: array[array[int]], 处理后的12×12二值向量列表
    """
    # 数据列表长度
    num = len(pixels_list)
    # 初始化结果列表
    feats_12x12 = np.empty((num, 12 ** 2))

    # 数据下尺寸至(1, 144)
    for i in range(num):
        og_img = minmax_scale(pixels_list[i], feature_range=(0, 1))
        # og_img = np.reshape(og_img, (28, 28))
        img = Image.fromarray(np.uint8(np.round(og_img)))
        # NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX, HAMMING
        img = img.resize((12, 12), Image.BILINEAR)
        img = np.array(img)
        feats_12x12[i] = np.reshape(img, (1, 12 * 12))

    return feats_12x12


def data_pretreat():
    """
    数据预处理，以矩阵文件形式保存

    Args:

    Returns:

    """
    global train_list_path, test_list_path, train_feature_path, train_label_path, result_list_path, number_select

    # 训练集实际数据路径
    train_list = os.listdir(train_list_path)
    # 导入并集成数据
    label_list = []
    pixels_list = []
    for file_name in train_list:
        serial_number, label = label_get(file_name)
        label_list.append(label)
        file_path = train_list_path + file_name
        pixels = image_read(file_path)
        pixels_list.append(pixels)

    # 将像素列表转换为NumPy数组并处理
    pixels_list = np.array(pixels_list)
    label_list = np.array(label_list)
    pixels_list = create_12x12_binary(pixels_list, label_list)

    # 将数据保存为二进制文件
    np.save(train_feature_path, pixels_list)
    np.save(train_label_path, label_list)

    # 测试集实际数据路径
    test_list = os.listdir(test_list_path)
    # 导入并集成数据
    label_list = []
    pixels_list = []
    for file_name in test_list:
        serial_number, label = label_get(file_name)
        label_list.append(label)
        file_path = test_list_path + file_name
        pixels = image_read(file_path)
        pixels_list.append(pixels)

    # 将像素列表转换为NumPy数组并处理
    pixels_list = np.array(pixels_list)
    label_list = np.array(label_list)
    pixels_list = create_12x12_binary(pixels_list, label_list)

    for i in range(len(number_select)):
        path = result_list_path + "true_" + str(number_select[i]) + "_" + str(label_list[number_select[i]]) + ".png"
        display(pixels_list[number_select[i]], path)

    return pixels_list, label_list
