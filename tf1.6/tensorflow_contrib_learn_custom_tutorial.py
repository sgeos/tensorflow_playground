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

def repl_block( expression, globals=None, locals=None ):
  prompt = ">>>"
  for line in expression.splitlines():
    print(f"{prompt} {line}")
    prompt = "..."
  eval(compile(expression + "\n", "<string>", "single"), globals, locals)

code = """
  import tensorflow as tf
  import numpy as np
"""
repl(code.strip(), globals(), locals())
print("")

code = """
def model(features, labels, mode):
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  loss = tf.reduce_sum(tf.square(y - labels))
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
  estimator = tf.contrib.learn.Estimator(model_fn=model)

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

