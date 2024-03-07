import time

current_time = time.time()

import sys
sys.setrecursionlimit(100000000)

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

ds, metadata = tfds.load('mnist', data_dir="./Datasets/", as_supervised=True, with_info=True, shuffle_files=True)
# assert isinstance(ds, tf.data.Dataset)
# print(ds)

trainer_data = ds['train']

names_of_clases = metadata.features['label'].names

# print("names of the classes" + str(names_of_clases))

def normalize(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

trainer_data = trainer_data.map(normalize)

# for image, label in trainer_data.take(1):
#   break
# image = image.numpy().reshape((28, 28))

# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(trainer_data.take(25)):
#   image = image.numpy().reshape((28, 28))
#   plt.subplot(5, 5, i + 1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(image, cmap=plt.cm.binary)
#   plt.xlabel(names_of_clases[label])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),


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

history = model.fit(trainer_data, epochs=50, steps_per_epoch=math.ceil(metadata.splits['train'].num_examples/SIZE_OF_BATCH))

plt.xlabel("# Epoch")
plt.ylabel("# Lost magnitude")
plt.plot(history.history["loss"])

model.save("ModelNumbers.h5")

#En el terminal
# !pip install tensorflowjs
# !mkdir tfjs_target_dir
# !tensorflowjs_converter --input_format keras ModelNumbers.h5 tfjs_target_dir
# !ls

print("The program has been runned in " + str(time.time() - current_time) + "s")
