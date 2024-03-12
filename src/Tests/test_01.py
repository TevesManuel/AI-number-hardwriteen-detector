import time

init_time = time.time()

print("Loading libraries...")
print("0/2 loaded")
import tensorflow as tf
print("1/2 loaded")
import tensorflow_datasets as tfds
print("2/2 loaded")
from keras.models import load_model

ds = tfds.load('mnist', split="test", as_supervised=True)

ds = ds.cache() #Pass the dataset to the memory for more velocity

#ds[0] are images
#ds[1] are labels

model = load_model("./ModelNumbers.h5")

results = []
i = 0
for image, label in ds:
    predict = model.predict([tf.reshape(image, [-1, 28, 28, 1])])
    # print(tf.argmax(predict, axis=1).numpy()[0])
    # print("LABEL:" + str(label.numpy()))
    # print(tf.argmax(predict, axis=1).numpy()[0] == label.numpy())
    results.append(tf.argmax(predict, axis=1).numpy()[0] == label.numpy())
    print(str(i) + "/" + str(len(ds)))
    i += 1

ok = 0
for i in results:
    if i == True:
        ok+=1
print(len(results))
print(ok)
print("Acuraccy: " + str(ok/len(results)*100) + "%")
print("Libraries has been loaded in " + str(time.time() - init_time) + "s")
print("The test has been runned in " + str(time.time() - init_time) + "s")