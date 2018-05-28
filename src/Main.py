from src.UNetKeras import *
import tensorflow as tf
X, y = get_data()
#
model = UNetKeras(high=512, weight=512, chanel=3)
model.compile()
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint("model/model.h5", monitor="val_loss", verbose=1,
                                                     save_best_only=True)
model.fit(X, y, batch_size=1, epochs=50, validation_split=0.01, callbacks=[ModelCheckpoint])
#
# model.model.predict()

print(X.shape, y.shape)
print(y[:, :, :, 0])
