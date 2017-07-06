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

  node1 = tf.constant(3.0, name='node1', dtype=tf.float32)
  node2 = tf.constant(4.0, name='node2')
  print(node1, node2)
  sess = tf.Session()
  print(sess.run([node1, node2]))
  node3 = tf.add(node1, node2, name='node3')
  print(f"node3: {node3}")
  print(f"sess.run(node3): {sess.run(node3)}")
  a = tf.placeholder(tf.float32, name='a_node')
  b = tf.placeholder(tf.float32, name='b_node')
  adder_node = tf.add(a, b, name='adder_node')
  print(sess.run(adder_node, {a: 3, b: 4.5}))
  print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
  add_and_triple = tf.multiply(adder_node, 3, name='add_and_triple')
  print(sess.run(add_and_triple, {a: 3, b: 4.5}))

  W = tf.Variable([.3], dtype=tf.float32, name='W')
  b = tf.Variable([-.3], dtype=tf.float32, name='b')
  x = tf.placeholder(tf.float32, shape=(4,), name='x')
  linear_model = tf.add(W * x, b, name='linear_model')
  init = tf.global_variables_initializer()
  sess.run(init)
  print(sess.run(linear_model, {x: [1,2,3,4]}))
  y = tf.placeholder(tf.float32, shape=(4,), name='y')
  squared_deltas = tf.square(linear_model - y, name='squared_deltas')
  loss = tf.reduce_sum(squared_deltas, name='loss')
  print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
  fixW = tf.assign(W, [-1.])
  fixb = tf.assign(b, [1.])
  sess.run([fixW, fixb])
  print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))

  optimizer = tf.train.GradientDescentOptimizer(0.01, name='optimizer')
  train = optimizer.minimize(loss, name='train')
  sess.run(init)
  summary_writer = tf.summary.FileWriter('log_tutorial', sess.graph)
  for value in [W, b, x, y]: tf.summary.scalar(value.op.name, tf.unstack(value)[0])
  summaries = tf.summary.merge_all()
  parameters = {x: [1,2,3,4], y: [0,-1,-2,-3]}
  for i in range(1000): summary_writer.add_summary(sess.run(summaries, parameters), i); sess.run(train, parameters)
  summary_writer.close()

  for line in ['Now, at the command line, we can start up TensorBoard.', '$ tensorboard --logdir=log_tutorial']: print(line)
  for line in ['TensorBoard runs as a local web app, on port 6006.', '$ open http://localhost:6006/#graphs', '$ open http://localhost:6006/#events']: print(line)
"""
repl(code.strip(), globals(), locals())

