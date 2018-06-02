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

X, y = get_data()
X, y = shuffle(X, y, random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2018)
del X, y
model = UNetKeras()

model.model.save()

model.compile()
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint("model/model.h5", monitor="val_loss", verbose=1,
                                                     save_best_only=True)
model.fit(X_train, y_train, batch_size=1, epochs=10, validation_split=0.01, callbacks=[ModelCheckpoint])

pred = model.predict(X_test, y_test)
