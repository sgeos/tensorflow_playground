#!/usr/bin/env python

# Reference:
#   https://www.tensorflow.org/get_started/tflearn

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
  import tensorflow as tf
  import numpy as np

  IRIS_HOST = "http://download.tensorflow.org/data"
  IRIS_LOCAL_PATH = "iris"
  IRIS_TRAINING = "iris_training.csv"
  IRIS_TEST = "iris_test.csv"
  IRIS_TRAINING_URL = f"{IRIS_HOST}/{IRIS_TRAINING}"
  IRIS_TEST_URL = f"{IRIS_HOST}/{IRIS_TEST}"
  IRIS_TRAINING_FILE = f"{IRIS_LOCAL_PATH}/{IRIS_TRAINING}"
  IRIS_TEST_FILE = f"{IRIS_LOCAL_PATH}/{IRIS_TEST}"
"""
repl(code.strip(), globals(), locals())
print('')

code = """
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)
  return x, y
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
def get_test_inputs():
  x = tf.constant(test_set.data)
  y = tf.constant(test_set.target)
  return x, y
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
def new_samples():
  return np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
if not os.path.exists(IRIS_LOCAL_PATH):
  os.mkdir(IRIS_LOCAL_PATH)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
if not os.path.exists(IRIS_TRAINING_FILE):
  raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
  with open(IRIS_TRAINING_FILE, "wb") as f:
    f.write(raw)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
if not os.path.exists(IRIS_TEST_FILE):
  raw = urllib.request.urlopen(IRIS_TEST_URL).read()
  with open(IRIS_TEST_FILE, "wb") as f:
    f.write(raw)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING_FILE, target_dtype=np.int, features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST_FILE, target_dtype=np.int, features_dtype=np.float32)
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="/tmp/iris_model")
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
  predictions = list(classifier.predict(input_fn=new_samples))
  print(''); print("Test Accuracy: {0:f}".format(accuracy_score)); print("New Samples, Class Predictions:    {}".format(predictions))
"""
repl(code.strip(), globals(), locals())
print('')

unused_code = """
  print("Unused code goes here.")
"""

