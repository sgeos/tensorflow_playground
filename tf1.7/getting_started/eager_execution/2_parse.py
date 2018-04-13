from __future__ import absolute_import, division, print_function

import os
import subprocess
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Reference:
#   https://www.tensorflow.org/get_started/eager

def enable_eager_execution():
  tf.enable_eager_execution()
  print("TensorFlow version: {}".format(tf.VERSION))
  print("Eager execution: {}".format(tf.executing_eagerly()))

def downlaod_data(train_dataset_url):
  train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url
  )
  print("Local copy of the dataset file: {}".format(train_dataset_fp))
  return train_dataset_fp

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

def main():
  enable_eager_execution()

  train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
  train_dataset_fp = downlaod_data(train_dataset_url)
  train_dataset = tf.data.TextLineDataset(train_dataset_fp)
  train_dataset = train_dataset.skip(1)             # skip the first header row
  train_dataset = train_dataset.map(parse_csv)      # parse each row
  train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
  train_dataset = train_dataset.batch(32)

  # View a single example entry from a batch
  features, label = tfe.Iterator(train_dataset).next()
  print("example features:", features[0])
  print("example label:", label[0])

main()

