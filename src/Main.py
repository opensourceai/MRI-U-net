#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author:ZERO
# E-mail:zero@osai.club
# Create Date: 2018年5月24日


from src.UNetKeras import get_data, UNetKeras, split_train_test
import tensorflow as tf

X, y = get_data("data")
X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.01, random_state=2018)
del X, y
model = UNetKeras(high=512, weight=512, chanel=3)

model.compile()
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint("model/model.h5", monitor="val_loss", verbose=1,
                                                     save_best_only=True)
model.fit(X_train, y_train, batch_size=1, epochs=50, validation_split=0.01, callbacks=[ModelCheckpoint])

pred = model.predict(X_test, y_test)
