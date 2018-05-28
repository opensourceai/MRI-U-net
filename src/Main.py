

from src.UNetKeras import *
import tensorflow as tf

model = UNetKeras(high=512, weight=512, chanel=3)
X, y = get_data("data")
model.compile()
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint("model/model.h5", monitor="val_loss", verbose=1,
                                                     save_best_only=True)
model.fit(X, y, batch_size=1, epochs=50, validation_split=0.01, callbacks=[ModelCheckpoint])

model.model.predict()
