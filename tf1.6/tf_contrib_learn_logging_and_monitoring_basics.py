#!/usr/bin/env python

# Reference:
#   https://www.tensorflow.org/get_started/input_fn
#   https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/input_fn/boston.py

def repl( expression, globals=None, locals=None ):
  for expression in expression.splitlines():
    expression = expression.strip()
    if expression:
      print(f">>> {expression}")
      eval(compile(expression + "\n", "<string>", "single"), globals, locals)
    else:
      print("")

def repl_block( expression, globals=None, locals=None ):
  prompt = ">>>"
  for line in expression.splitlines():
    print(f"{prompt} {line}")
    prompt = "..."
  eval(compile(expression + "\n", "<string>", "single"), globals, locals)

code = """
  import os
  import urllib.request
  import numpy as np
  import tensorflow as tf

  tf.logging.set_verbosity(tf.logging.INFO)

  DATA_HOST = "http://download.tensorflow.org/data"
  DATA_LOCAL_PATH = "data_iris"
  DATA_TRAIN = "iris_training.csv"
  DATA_TEST = "iris_test.csv"
"""
repl(code.strip(), globals(), locals())
print('')

code = """
if not os.path.exists(DATA_LOCAL_PATH):
  os.mkdir(DATA_LOCAL_PATH)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
for base_file in [DATA_TRAIN, DATA_TEST]:
  local_file = f"{DATA_LOCAL_PATH}/{base_file}"
  if not os.path.exists(local_file):
    remote_file = f"{DATA_HOST}/{base_file}"
    raw = urllib.request.urlopen(remote_file).read()
    with open(local_file, "wb") as f:
      f.write(raw)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
def input_fn(data_set):
  x = tf.constant(data_set.data)
  y = tf.constant(data_set.target)
  return x, y
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
def new_samples():
  return np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=f"{DATA_LOCAL_PATH}/{DATA_TRAIN}", target_dtype=np.int, features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=f"{DATA_LOCAL_PATH}/{DATA_TEST}", target_dtype=np.int, features_dtype=np.float32)

  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
"""
repl(code.strip(), globals(), locals())

code = """
  validation_metrics = {
    "accuracy": tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy, prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "precision": tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_precision, prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall": tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_recall, prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
  }
"""
repl_block(code.strip(), globals(), locals())

code = """
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_set.data, test_set.target, every_n_steps=50, metrics=validation_metrics, early_stopping_metric="loss", early_stopping_metric_minimize=True, early_stopping_rounds=200)
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir=f"{DATA_LOCAL_PATH}/model", config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
  classifier.fit(input_fn=lambda: input_fn(training_set), steps=2000, monitors=[validation_monitor])

  accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)["accuracy"]
  predictions = list(classifier.predict(input_fn=new_samples))

  print('Accuracy: {0:f}'.format(accuracy_score)) ; print('Predictions: {}'.format(str(predictions)))
"""
repl(code.strip(), globals(), locals())

unused_code = """
  accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
  new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
  y = list(classifier.predict(new_samples, as_iterable=True))
  classifier.fit(x=training_set.data, y=training_set.target, steps=2000)
  print("Unused code goes here.")
"""

