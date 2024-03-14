#Converter script
import time

init_time = time.time()

print("Loading libraries...")
print("0/2")
import tensorflow as tf
print("1/2")
from tensorflowjs import tfjs
print("2/2")
print("Libraries has been loaded in " + str(time.time() - init_time) + "s")

# Cargar el modelo Keras
model = tf.keras.models.load_model('ModelNumbers.h5')

# Convertir el modelo
converter = tfjs.converters.SavedModelConverter(model)
converted_model = converter.convert()

# (Opcional) Guardar el modelo convertido
bundled_model = tfjs.converters.bundle_into_web_bundle(converted_model)

with open('./tfjs/ModelNumbers.js', 'wb') as f:
    f.write(bundled_model)

print("The script has been runned in " + str(time.time() - init_time) + "s")