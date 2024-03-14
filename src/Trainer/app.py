#Batch is a Data Set, in this case each batch is 32 data
BATCH_SIZE = 32
#Epochs of the training
TRAINING_EPOCHS = 50

import time

init_time = time.time()

import sys
#32 Bit max C INT value ( for greater compatibility between systems )
sys.setrecursionlimit(2**31 - 1)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'#Disable all warnings, except FATAL ERR

count_libraries = 6
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
# from  import ImageDataGenerator
print("5/" + str(count_libraries) + " loaded")
from tqdm import tqdm
print("6/" + str(count_libraries) + " loaded")

print("Libraries has been loaded in " + str(time.time() - init_time) + "s")

#Verify the correct struct & pass to value between 0.0 & 1.0 for more efficiency
def normalize(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

print("Getting dataset...")

#Struct of the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", name="conv1"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", name="conv2"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation='softmax'),
])

#Construct the neural network
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#tfs.load flags
#data_dir is the dir where the dataset are downloaded
#as_supervised is for the ds give image and label
#with_info is for the function return metadata too
#shuffle_files is for shufle data
ds, metadata = tfds.load('mnist', data_dir="./Datasets/", as_supervised=True, with_info=True, shuffle_files=True)

print("Dataset has downloaded.")

trainer_data = ds['train']
trainer_data = trainer_data.cache() #Pass to the cache for more velocity

#Geting names of the clasess
names_of_clases = metadata.features['label'].names

trainer_data = trainer_data.map(normalize)

print("passing dataset to np array")

#Convert Dataset to np Array
trainer_data_images_np = []
trainer_data_labels_np = []

for image, label in tqdm(trainer_data, total=len(trainer_data), desc="Converting to numpy", colour='#00ff00', ncols=100):
  trainer_data_images_np.append(image.numpy())
  trainer_data_labels_np.append(label.numpy())
trainer_data_images_np = np.array(trainer_data_images_np)
trainer_data_labels_np = np.array(trainer_data_labels_np)

print("Dataset has been converted correctly.")

# Define data generator with my flags
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,  # Range of the random rotation
    width_shift_range=0.25,  # Desplazamiento horizontal aleatorio
    height_shift_range=0.25,  # Desplazamiento vertical aleatorio
    shear_range=0.1,  # Estiramiento aleatorio
    zoom_range=[0.5, 1.5],  # Rango de zoom aleatorio
    horizontal_flip=False,  # Volteo horizontal aleatorio
    vertical_flip=False,  # Volteo vertical aleatorio
    fill_mode='nearest'  # Método de relleno para píxeles nuevos generados
)

#Make some changes over images
datagen.fit(trainer_data_images_np)

# Apply transforms and generate batchs
trainer_data = datagen.flow(
    trainer_data_images_np,
    trainer_data_labels_np,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print("DS LEN " + str(len(trainer_data)))

history = model.fit(
                    trainer_data,
                    epochs=TRAINING_EPOCHS,
                    steps_per_epoch=int(np.ceil(len(trainer_data)))
                   )


model.save("ModelNumbers.h5")

print("The program has been runned in " + str(time.time() - init_time) + "s")

plt.xlabel("# Epoch")
plt.ylabel("# Lost magnitude")
plt.plot(history.history["loss"])
plt.show()