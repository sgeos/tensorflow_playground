#!/usr/bin/env python

# Reference
#   https://www.tensorflow.org/get_started/get_started

def repl( expression, globals=None, locals=None ):
  for expression in expression.splitlines():
    expression = expression.strip()
    if expression:
      print(f">>> {expression}")
      eval(compile(expression + "\n", "<string>", "single"), globals, locals)
    else:
      print("")

code = """
  import tensorflow as tf
  import numpy as np

  features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
  estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

  x_train = np.array([1., 2., 3., 4.])
  y_train = np.array([0., -1., -2., -3.])
  x_eval = np.array([2., 5., 8., 1.])
  y_eval = np.array([-1.01, -4.1, -7., 0.])
  input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000)
  eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000)

  estimator.fit(input_fn=input_fn, steps=1000)
  train_loss = estimator.evaluate(input_fn=input_fn)
  eval_loss = estimator.evaluate(input_fn=eval_input_fn)

  print(f"train loss: {train_loss['loss']} (global step {train_loss['global_step']})") ; print(f"eval loss:  {eval_loss['loss']} (global step {eval_loss['global_step']})")
"""
repl(code.strip(), globals(), locals())

