import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import cv2
# import matplotlib.pyplot as plt
import h5py
from scipy import ndimage
import numpy as np


class UNetKeras(object):

    def __init__(self, high=512, weight=512, chanel=1):
        inputs = tf.keras.layers.Input((high, weight, chanel))
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            pool4)
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            conv5)
        drop5 = tf.keras.layers.Dropout(0.5)(conv5)

        up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = tf.keras.layers.Concatenate(axis=3)([drop4, up6])
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = tf.keras.layers.Concatenate(axis=3)([conv3, up7])
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = tf.keras.layers.Concatenate(axis=3)([conv2, up8])
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = tf.keras.layers.Concatenate(axis=3)([conv1, up9])
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # 最后一层为softmax层，输出的通道为3，对应着背景、膀胱壁区域、肿瘤区域
        conv10 = tf.keras.layers.Conv2D(3, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(
            conv9)

        self.model = tf.keras.Model(inputs=inputs, outputs=conv10)

    def compile(self, optimizer=tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.90, beta_2=0.90),
                loss="categorical_crossentropy",
                metrics=['accuracy']):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def fit(self, x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None, ):
        self.model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                       class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps)


# 加载数据
def _load():
    """
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
        with h5py.File('../data/images.h5', 'r') as file:
            images = file.get("images")
            images = np.array(images, dtype=np.float32)

    if not os.path.exists("../data/labels.h5"):
        labels = []
        for i in range(1, 2201):
            now_file_path = "../data/Label/Label" + str(i) + ".png"
            label = np.array(ndimage.imread(now_file_path, flatten=False))
            labels.append(label)  # images shape=(m,64,64,3)
        labels = np.array(labels, copy=True)
        file = h5py.File('../data/labels.h5', 'w')  # 创建HDF5文件
        file.create_dataset('labels', data=labels)  # 写入
        file.close()

    else:
        with h5py.File('../data/labels.h5', 'r') as file:
            labels = file.get("labels")
            labels = np.array(labels, dtype=np.float32)

    images /= 255
    train_image = np.expand_dims(images[:], -1)
    train_label = labels[:]
    del images, labels
    return train_image, train_label


def get_data():

    image, label = _load()
    # 0 为背景区域
    label[label == 128] = 1  # 膀胱壁区域
    label[label == 255] = 2  # 肿瘤区域

    # one_hot 处理
    # (n,512,512) => (n,512,512,1)
    label = np.expand_dims(label, -1)
    print(label.shape)
    label_tile = np.tile(label, (1, 1, 1, 3))
    label_tile[:, :, :, 0] = np.where(label_tile[:, :, :, 0] == 0, 1, 0)
    label_tile[:, :, :, 1] = np.where(label_tile[:, :, :, 1] == 1, 1, 0)
    label_tile[:, :, :, 2] = np.where(label_tile[:, :, :, 2] == 2, 1, 0)

    label = label_tile
    return image, label


# 划分训练集和测试集 可选步骤
def split_train_test(X, y, test_size=0.33, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test


# 打乱数据顺序 可选步骤
def shuffle_data(X, y, random_state=0, n_samples=2):
    X, y = shuffle(X, y, random_state, n_samples)
    return X, y
