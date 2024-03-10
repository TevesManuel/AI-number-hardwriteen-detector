#Summary test

import tensorflow as tf

model = tf.keras.models.load_model("./../../ModelNumbers.h5")

model.summary()