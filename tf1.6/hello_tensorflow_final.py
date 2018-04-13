#!/usr/bin/env python

# Reference:
#   https://www.oreilly.com/learning/hello-tensorflow

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

  x = tf.constant(1.0, name='input')
  w = tf.Variable(0.8, name='weight')
  y = tf.multiply(w, x, name='output')
  y_target = tf.constant(0.0, name='target_output')
  loss = tf.pow(y - y_target, 2, name='loss')
  train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

  for value in [x, w, y, y_target, loss]: tf.summary.scalar(value.op.name, value)
  summaries = tf.summary.merge_all()

  sess = tf.Session()
  summary_writer = tf.summary.FileWriter('log_final_simple_stats', sess.graph)

  sess.run(tf.global_variables_initializer())
  for i in range(100): summary_writer.add_summary(sess.run(summaries), i); sess.run(train_step)
  summary_writer.close()

  for line in ['Now, at the command line, we can start up TensorBoard.', '$ tensorboard --logdir=log_final_simple_stats']: print(line)
  for line in ['TensorBoard runs as a local web app, on port 6006.', '$ open http://localhost:6006/#graphs', '$ open http://localhost:6006/#events']: print(line)
"""
repl(code.strip(), globals(), locals())

