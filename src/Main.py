#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author:ZERO
# E-mail:zero@osai.club
# Create Date: 2018年5月24日

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.utils import get_data
from src.UNetKeras import UNetKeras
import tensorflow as tf

if __name__ == "__main__":
    print("=========          Get data            =========")
    X, y = get_data()
    # X, y = shuffle(X, y, random_state=2018)
    print("=========Split train sets and test sets=========")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2018)
    del X, y

    print("=========          Build model         =========")
    model = UNetKeras()
    model.compile()

    print("=========       Start train model      =========")
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint("model/val_best_model.h5", monitor="val_loss", verbose=1,
                                                         save_best_only=True)
    model.fit(X_train, y_train, batch_size=1, epochs=10, validation_split=0.01, callbacks=[ModelCheckpoint])
    print("=========    Save last time model      =========")

    model.model.save("model/model_final_time.h5")

    print("=========    Model evaluate Start      =========")
    print("=========  Test the last saved model   =========")
    model.model.evaluate(X_test, y_test, batch_size=32)
    pred = model.predict(X_test)
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    images_test = np.reshape(np.argmax(y_test, axis=-1), (-1, 512, 512))
    images_pred = np.reshape(pred, (-1, 512, 512))
    predict_file_path = ".predict_images_last_saved_model/"
    if not os.path.exists(predict_file_path):
        os.mkdir(predict_file_path)

    for i in range(len(images_pred)):
        # plt.subplot(1, 2, 1)
        # plt.imshow(images_pred[i], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow(images_test[i], cmap="gray")
        plt.imsave(predict_file_path + "" + str(i) + "p.png", images_pred[i])
        plt.imsave(predict_file_path + str(i) + "t.png", images_test[i])
    del model, pred
    print("=========   Test the val best model    =========")

    predict_file_path = ".predict_images_val_best_model/"
    if not os.path.exists(predict_file_path):
        os.mkdir(predict_file_path)

    val_best_model = tf.keras.models.load_model("model/val_best_model.h5")
    val_best_model.evaluate(X_test, y_test, batch_size=32)
    pred = val_best_model.predict(X_test)
    images_pred = np.reshape(np.argmax(pred, axis=-1), (-1, 512, 512))

    for i in range(len(images_pred)):
        # plt.subplot(1, 2, 1)
        # plt.imshow(images_pred[i], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow(images_test[i], cmap="gray")
        plt.imsave(predict_file_path + str(i) + "p.png", images_pred[i])
        plt.imsave(predict_file_path + str(i) + "t.png", images_test[i])
