import time

current_time = time.time()

import sys
#32 Bit max C INT value ( for greater compatibility between systems )
sys.setrecursionlimit(2**31 - 1)

count_libraries = 4
print("Loading libraries...")
print("0/" + str(count_libraries) + " loaded")
import matplotlib.pyplot as plt
print("1/" + str(count_libraries) + " loaded")
import numpy as np
print("2/" + str(count_libraries) + " loaded")
import tensorflow as tf
print("3/" + str(count_libraries) + " loaded")
import tensorflow_datasets as tfds
print("4/" + str(count_libraries) + " loaded")
print("Libraries has been loaded in " + str(time.time() - current_time) + "s")

print("Getting dataset...")
ds, metadata = tfds.load('mnist', data_dir="./Datasets/", as_supervised=True, with_info=True, shuffle_files=True)
# assert isinstance(ds, tf.data.Dataset)
# print(ds)
print("Dataset has downloaded.")

trainer_data = ds['train']

names_of_clases = metadata.features['label'].names

# print("names of the classes" + str(names_of_clases))

def normalize(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

trainer_data = trainer_data.map(normalize)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", name="conv1"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", name="conv2"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),

    # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # tf.keras.layers.Dense(50, activation=tf.nn.relu),
    # tf.keras.layers.Dense(10, activation=tf.nn.relu),
    # tf.keras.layers.Dense(10, activation=tf.nn.softma x),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

import math

SIZE_OF_BATCH = 32

trainer_data = trainer_data.repeat().shuffle(metadata.splits['train'].num_examples).batch(SIZE_OF_BATCH)

history = model.fit(trainer_data, epochs=5, steps_per_epoch=math.ceil(metadata.splits['train'].num_examples/SIZE_OF_BATCH))

plt.xlabel("# Epoch")
plt.ylabel("# Lost magnitude")
plt.plot(history.history["loss"])
plt.show()

model.save("ModelNumbers.h5")

#En el terminal
# !pip install tensorflowjs
# !mkdir tfjs_target_dir
# !tensorflowjs_converter --input_format keras ModelNumbers.h5 tfjs_target_dir
# !ls

print("The program has been runned in " + str(time.time() - current_time) + "s")
