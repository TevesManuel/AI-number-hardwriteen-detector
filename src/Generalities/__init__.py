#Batch is a Data Set, in this case each batch is 32 data
BATCH_SIZE = 32

import tensorflow as tf
#Verify the correct struct & pass to value between 0.0 & 1.0 for more efficiency
def normalize(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label