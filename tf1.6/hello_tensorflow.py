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

  graph = tf.get_default_graph()
  input_value = tf.constant(1.0, name='input_value')
  graph.get_operations()
  graph.get_operations()[0].node_def
  input_value
  sess = tf.Session()
  sess.run(input_value)
  weight = tf.Variable(0.8, name='weight_variable')
  for op in graph.get_operations(): print(op.name)
  output_value = weight * input_value
  op = graph.get_operations()[-1]
  op.name
  for op_input in op.inputs: print(op_input)
  init = tf.global_variables_initializer()
  sess.run(init)
  sess.run(output_value)

  x = tf.constant(1.0, name='input')
  w = tf.Variable(0.8, name='weight')
  y = tf.multiply(w, x, name='output')
  summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)
  summary_writer.close()
  for line in ['Now, at the command line, we can start up TensorBoard. TensorBoard runs as a local web app, on port 6006. ', '$ tensorboard --logdir=log_simple_graph', '$ open http://localhost:6006/#graphs']: print(line)

  y_target = tf.constant(0.0, name='y_target')
  loss = (y - y_target)**2
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.025)
  gradients_and_variables = optimizer.compute_gradients(loss)
  sess.run(tf.global_variables_initializer())
  sess.run(gradients_and_variables[1][0])
  sess.run(optimizer.apply_gradients(gradients_and_variables))
  sess.run(w)
  train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
  for i in range(100): sess.run(train_step)
  sess.run(y)

  sess.run(tf.global_variables_initializer())
  for i in range(100): print(f"before step {i}, y is {sess.run(y)}"); sess.run(train_step)

  summary_y = tf.summary.scalar('output', y)
  summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)
  sess.run(tf.global_variables_initializer())
  for i in range(100): summary_str = sess.run(summary_y); summary_writer.add_summary(summary_str, i); sess.run(train_step)
  summary_writer.close()
  for line in ['Now, at the command line, we can start up TensorBoard. TensorBoard runs as a local web app, on port 6006. ', '$ tensorboard --logdir=log_simple_stats', '$ open http://localhost:6006/#events']: print(line)
"""
repl(code.strip(), globals(), locals())

