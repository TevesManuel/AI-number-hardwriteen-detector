#Summary test
import time

init_time = time.time()

print("Loading libraries...")
print("0/1")
import tensorflow as tf
print("1/1")
print("Libraries has been loaded in " + str(time.time() - init_time) + "s")

model = tf.keras.models.load_model("./ModelNumbers.h5")

#Automatly print the struct of the loaded model
model.summary()

print("The program has been runned in " + str(time.time() - init_time) + "s")