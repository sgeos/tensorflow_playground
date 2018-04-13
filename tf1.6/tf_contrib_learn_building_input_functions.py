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
  import itertools
  import pandas as pd
  import tensorflow as tf

  DATA_HOST = "http://download.tensorflow.org/data"
  DATA_LOCAL_PATH = "data_boston"
  DATA_TRAIN = "boston_train.csv"
  DATA_TEST = "boston_test.csv"
  DATA_PREDICT = "boston_predict.csv"
"""
repl(code.strip(), globals(), locals())
print('')

code = """
def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
if not os.path.exists(DATA_LOCAL_PATH):
  os.mkdir(DATA_LOCAL_PATH)
"""
repl_block(code.strip(), globals(), locals())
print('')

code = """
for base_file in [DATA_TRAIN, DATA_TEST, DATA_PREDICT]:
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
  tf.logging.set_verbosity(tf.logging.INFO)
  LABEL = "medv"
  COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
  FEATURES = [column for column in COLUMNS if column != LABEL]

  training_set = pd.read_csv(f"{DATA_LOCAL_PATH}/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
  test_set = pd.read_csv(f"{DATA_LOCAL_PATH}/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
  prediction_set = pd.read_csv(f"{DATA_LOCAL_PATH}/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

  feature_cols = [tf.contrib.layers.real_valued_column(feature) for feature in FEATURES]
  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10], model_dir=f"{DATA_LOCAL_PATH}/model")
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  loss_score = ev["loss"]

  y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
  predictions = list(itertools.islice(y, 6))

  print("Loss: {0:f}".format(loss_score)) ; print ("Predictions: {}".format(str(predictions)))
"""
repl(code.strip(), globals(), locals())

unused_code = """
  print("Unused code goes here.")
"""

