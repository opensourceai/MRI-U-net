#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author:ZERO
# E-mail:zero@osai.club
# Create Date: 2018年5月30日

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import h5py
from scipy import ndimage


# 加载数据
def _load(path_name):
    """

    :param path_name:
    :return:
    """
    import os
    if not os.path.exists("../data/images.h5"):

        images = []
        for i in range(1, 2201):
            now_file_path = "../data/Image/IM" + str(i) + ".png"
            image = np.array(ndimage.imread(now_file_path, flatten=False))
            images.append(image)  # images shape=(m,64,64,3)
        images = np.array(images, copy=True)
        file = h5py.File('../data/images.h5', 'w')  # 创建HDF5文件
        file.create_dataset('images', data=images)  # 写入
        file.close()
    else:
        with h5py.File(path_name + '/images.h5', 'r') as flie:
            images = flie.get("images")
            images = np.array(images, dtype=np.float32)

    if not os.path.exists("../data/labels.h5"):
        labels = []
        for i in range(1, 2201):
            now_file_path = "Label/Label" + str(i) + ".png"
            label = np.array(ndimage.imread(now_file_path, flatten=False))
            labels.append(label)  # images shape=(m,64,64,3)
        labels = np.array(labels, copy=True)
        file = h5py.File('../data/labels.h5', 'w')  # 创建HDF5文件
        file.create_dataset('labels', data=labels)  # 写入
        file.close()

    else:
        with h5py.File(path_name + '/labels.h5', 'r') as flie:
            labels = flie.get("labels")
            labels = np.array(labels, dtype=np.float32)

    images /= 255
    train_image = np.expand_dims(images[:], -1)
    train_label = labels[:]
    del images, labels
    return train_image, train_label


def get_data(path_name):
    image, label = _load(path_name)
    label[label == 128] = 1  # 膀胱壁区域
    label[label == 255] = 2  # 肿瘤区域

    # one_hot 处理
    # label shape = (n,512,512,1)
    label = encode_one_hot(label, 3)

    return image, label


def encode_one_hot(x, classes_num=3):
    """
    :param array x: ,Single channel picture.
    :param int classes_num: Dimension of one_hot or Classification number.
    :return: label of one_hot
    :rtype: narray
    """
    if x.shape[-1] == 1:
        x_tiled = np.tile(x, (1, 1, 1, classes_num))

        for i in range(classes_num):
            x_tiled[:, :, :, i] = np.where(x_tiled[:, :, :, i] == i, 1, 0)

    else:
        raise IndexError("The last dimension is not 1")
    return x_tiled


# 划分训练集和测试集
def split_train_test(X, y, test_size=0.33, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test


# 打乱数据顺序
def shuffle_data(X, y, random_state=0, n_samples=None):
    X, y = shuffle(X, y, random_state, n_samples)
    return X, y
