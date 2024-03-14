import time

init_time = time.time()

print("Loading libraries...")
print("0/2 loaded")
import tensorflow as tf
print("1/2 loaded")
import tensorflow_datasets as tfds
print("2/2 loaded")
from keras.models import load_model
from tqdm import tqdm

ds = tfds.load('mnist', split="test", as_supervised=True, shuffle_files=True)

ds = ds.cache() #Pass the dataset to the memory for more velocity

#ds[0] are images
#ds[1] are labels

model = load_model("./ModelNumbers.h5")

num_samples = 100 # Max 10000

results = []
for image, label in tqdm(ds.take(num_samples), total=num_samples, desc="Testing...", colour='#00ff00', ncols=100):
    predict = model.predict([tf.reshape(image, [-1, 28, 28, 1])], verbose=0)
    # print(tf.argmax(predict, axis=1).numpy()[0]) # Result of the prediction
    # print("LABEL:" + str(label.numpy())) # Expectation result
    # print(tf.argmax(predict, axis=1).numpy()[0] == label.numpy()) # Comparassion beetween prediction and the expectation
    results.append(tf.argmax(predict, axis=1).numpy()[0] == label.numpy())


#Counting Trues in the comparassions
ok = 0
for i in results:
    if i == True:
        ok+=1

#Calculating acuraccy
acuraccy = ok/len(results)*100
# print(ok)
# print(len(results))

print("Acuraccy: " + str(acuraccy) + "%")

print("Libraries has been loaded in " + str(time.time() - init_time) + "s")
print("The test has been runned in " + str(time.time() - init_time) + "s")