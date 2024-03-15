import time

init_time = time.time()

import sys
sys.path.append('../')#Move to src directory for access to Generalities

from Generalities import normalize

print("Loading libraries...")
print("0/2 loaded")
import tensorflow as tf
print("1/2 loaded")
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
print("2/2 loaded")
from keras.models import load_model
from tqdm import tqdm
from Generalities import normalize
print("Libraries has been loaded in " + str(time.time() - init_time) + "s")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'#Disable all warnings, except FATAL ERR

ds, metadata = tfds.load('mnist', data_dir="./Datasets/", as_supervised=True, with_info=True, shuffle_files=True)

test_data = ds['test']
test_data = test_data.cache() #Pass the dataset to the memory for more velocity
test_data = test_data.map(normalize)

#ds[0] are images
#ds[1] are labels

print("Loaded " + str(len(test_data)) + " samples")

model = load_model("./ModelNumbers.h5")

num_samples = 100 # Max 10000

ok = 0
for image, label in tqdm(test_data.take(num_samples), total=num_samples, desc="Testing...", colour='#00ff00', ncols=100):
    predict = model.predict([tf.reshape(image, [1, 28, 28, 1])], verbose=0)
    # print(tf.argmax(predict, axis=1).numpy()[0]) # Result of the prediction
    # print("LABEL:" + str(label.numpy())) # Expectation result
    # print(tf.argmax(predict, axis=1).numpy()[0] == label.numpy()) # Comparassion beetween prediction and the expectation
    result = tf.argmax(predict, axis=1).numpy()[0]
    if result == label.numpy():
        ok+=1
    else:
        plt.title("Expected: " + str(label.numpy()) + " | receibed : " + str(result))
        plt.imshow(image)
        plt.show()

#Calculating acuraccy
acuraccy = ok/num_samples*100
print(ok)
print(num_samples)
print("Acuraccy: " + str(acuraccy) + "%")

print("The test has been runned in " + str(time.time() - init_time) + "s")