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

def downlaod_dataset(dataset_url):
  dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(dataset_url),
    origin=dataset_url
  )
  return dataset_fp

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

def prepare_dataset(dataset_url):
  dataset_fp = downlaod_dataset(dataset_url)
  dataset = tf.data.TextLineDataset(dataset_fp)
  dataset = dataset.skip(1)             # skip the first header row
  dataset = dataset.map(parse_csv)      # parse each row
  dataset = dataset.shuffle(buffer_size=1000)  # randomize
  dataset = dataset.batch(32)
  return dataset

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

def prepare_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3)
  ])
  return model

def train(model, train_dataset):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

  ## Note: Rerunning this cell uses the same model variables

  # keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  num_epochs = 201

  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in tfe.Iterator(train_dataset):
      # Optimize the model
      grads = grad(model, x, y)
      optimizer.apply_gradients(
        zip(grads, model.variables),
        global_step=tf.train.get_or_create_global_step()
      )

      # Track progress
      epoch_loss_avg(loss(model, x, y))  # add current batch loss
      # compare predicted label to actual label
      epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
      print(
        "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
          epoch,
          epoch_loss_avg.result(),
          epoch_accuracy.result()
        )
      )
  return train_loss_results, train_accuracy_results

def test(model, test_dataset):
  test_accuracy = tfe.metrics.Accuracy()

  for (x, y) in tfe.Iterator(test_dataset):
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

  print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
  return test_accuracy.result()

def plot_results(train_loss_results, train_accuracy_results):
  fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
  fig.suptitle('Training Metrics')

  axes[0].set_ylabel("Loss", fontsize=14)
  axes[0].plot(train_loss_results)

  axes[1].set_ylabel("Accuracy", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=14)
  axes[1].plot(train_accuracy_results)

  plt.show()

def predict(model, predict_dataset):
  class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

  predict_dataset = tf.convert_to_tensor(predict_dataset)
  predictions = model(predict_dataset)

  for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))

def main(train_dataset_url, test_dataset_url, predict_dataset):
  enable_eager_execution()
  train_dataset = prepare_dataset(train_dataset_url)
  model = prepare_model()
  train_loss_results, train_accuracy_results = train(model, train_dataset)
  # plot_results(train_loss_results, train_accuracy_results)
  test_dataset = prepare_dataset(test_dataset_url)
  test(model, test_dataset)
  predict(model, predict_dataset)

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
test_dataset_url = "http://download.tensorflow.org/data/iris_test.csv"
predict_dataset = [
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
  ]
main(train_dataset_url, test_dataset_url, predict_dataset)

