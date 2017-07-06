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

  W = tf.Variable([.3], dtype=tf.float32, name='W')
  b = tf.Variable([-.3], dtype=tf.float32, name='b')
  x = tf.placeholder(tf.float32, shape=(4,), name='x')
  linear_model = tf.add(W * x, b, name='linear_model')
  y = tf.placeholder(tf.float32, shape=(4,), name='y')
  loss = tf.reduce_sum(tf.square(linear_model - y), name='loss')
  optimizer = tf.train.GradientDescentOptimizer(0.01, name='optimizer')
  train = optimizer.minimize(loss, name='train')
  x_training = [1,2,3,4]
  y_training = [0,-1,-2,-3]
  training_parameters = {x: x_training, y: y_training}

  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  logdir = 'log_tutorial_final'
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)
  for value in [W, b, x, y]: tf.summary.scalar(value.op.name, tf.unstack(value)[0])
  summaries = tf.summary.merge_all()
  for i in range(1000): summary_writer.add_summary(sess.run(summaries, training_parameters), i); sess.run(train, training_parameters)
  summary_writer.close()

  final_W, final_b, final_loss = sess.run([W, b, loss], training_parameters)
  for line in [f"Final W:    {final_W}", f"Final b:    {final_b}", f"Final loss: {final_loss}"]: print(line)

  for line in ['Now, at the command line, we can start up TensorBoard.', f'$ tensorboard --logdir={logdir}']: print(line)
  for line in ['TensorBoard runs as a local web app, on port 6006.', '$ open http://localhost:6006/#graphs', '$ open http://localhost:6006/#events']: print(line)
"""
repl(code.strip(), globals(), locals())

