from __future__ import absolute_import, division, print_function

import os
import subprocess
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Reference:
#   https://www.tensorflow.org/get_started/eager

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))
command=['head','-n5', train_dataset_fp]
data = subprocess.check_output(command)
print('$ ' + ' '.join(command))
print(data.decode('utf8'))

