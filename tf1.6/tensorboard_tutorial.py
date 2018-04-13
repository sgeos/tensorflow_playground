#!/usr/bin/env python

# Reference
#   http://web.stanford.edu/class/cs20si/lectures/notes_02.pdf

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

  sess = tf.Session()

  my_const = tf.constant([1.0, 2.0], name="my_constant")
  print(tf.get_default_graph().as_graph_def())

  vector = tf.constant([2, 2], name="vector"); print(sess.run(vector))
  matrix = tf.constant([[0, 1], [2, 3]], name="vector"); print(sess.run(matrix))
  input_tensor = tf.constant([[0, 1], [2, 3], [4, 5]], name="input_tensor"); print(sess.run(input_tensor))
  zeros = tf.zeros([2, 3], tf.int32, name="zeros"); print(sess.run(zeros))
  zeros_like = tf.zeros_like(input_tensor, name="zeros_like"); print(sess.run(zeros_like))
  ones = tf.ones([2, 3], tf.int32, name="ones"); print(sess.run(ones))
  ones_like = tf.ones_like(input_tensor, name="ones_like"); print(sess.run(ones_like))
  fill = tf.fill([2, 3], 8, name="fill"); print(sess.run(fill))
  linspace = tf.linspace(10.1, 12.9, 5, name="linspace"); print(sess.run(linspace))
  range_a = tf.range(3, 18, 3, name="range_a"); print(sess.run(range_a))
  range_b = tf.range(3, 1, -0.5, name="range_b"); print(sess.run(range_b))
  range_c = tf.range(5, name="range_c"); print(sess.run(range_c))
  random_normal = tf.random_normal([2, 3], 0.0, 1.5, tf.float32, name="random_normal"); print(sess.run(random_normal))
  truncated_normal = tf.truncated_normal([2, 3], 0.0, 1.5, tf.float32, name="truncated_normal"); print(sess.run(truncated_normal))
  random_uniform = tf.random_uniform([2, 3], -1000, 1000, tf.int32, name="random_uniform"); print(sess.run(random_uniform))
  random_shuffle = tf.random_shuffle(tf.range(10), name="random_shuffle"); print(sess.run(random_shuffle))
  random_crop = tf.random_crop([random_shuffle, random_shuffle], [2, 3], name="random_crop"); print(sess.run(random_crop))
  multinomial = tf.multinomial(tf.log([[10.0, 10.0]]), 5, name="multinomial"); print(sess.run(multinomial))
  gamma_a = tf.random_gamma([4], [0.5, 1.0, 1.5], name="gamma_a"); print(sess.run(gamma_a))
  gamma_b = tf.random_gamma([3, 2], [3.5, 1.5], beta=[1.0, 2.0], name="gamma_b"); print(sess.run(gamma_b))

  a_scaler = tf.constant(2, name="a_scaler"); print(sess.run(a_scaler))
  b_scaler = tf.constant(3, name="b_scaler"); print(sess.run(b_scaler))
  a = tf.constant([3, 6], name="a"); print(sess.run(a))
  b = tf.constant([2, 2], name="b"); print(sess.run(b))
  add = tf.add(a, b, name="add"); print(sess.run(add))
  add_n = tf.add_n([a, b, b], name="add_n"); print(sess.run(add_n))
  multiply = tf.multiply(a, b, name="multiply"); print(sess.run(multiply))
  matrix_multiply = tf.matmul(tf.reshape(a, shape=[1,2]), tf.reshape(b, shape=[2,1]), name="matrix_multiply"); print(sess.run(matrix_multiply)); print("Error without reshaping!")
  divide = tf.div(a, b, name="divide"); print(sess.run(divide))
  remainder = tf.mod(a, b, name="remainder"); print(sess.run(remainder))

  scaler_variable = tf.Variable(2, name="scaler_variable"); sess.run(scaler_variable.initializer); print(sess.run(scaler_variable))
  vector_variable = tf.Variable([2, 3], name="vector_variable"); sess.run(vector_variable.initializer); print(sess.run(vector_variable))
  matrix_variable = tf.Variable([[0, 1], [2, 3]], name="matrix_variable"); sess.run(matrix_variable.initializer); print(sess.run(matrix_variable))
  zeros_variable = tf.Variable(tf.zeros([4, 16]), name="zeros_variable"); sess.run(zeros_variable.initializer); print(sess.run(zeros_variable))

  a_variable = tf.Variable(-1, name="a_variable")
  b_variable = tf.Variable([-2, -3], name="b_variable")
  init_ab_variables = tf.variables_initializer([a_variable, b_variable], name="init_ab_variables")
  sess.run(init_ab_variables)
  print(f"a: {sess.run(a_variable)}, b: {sess.run(b_variable)}")

  weights = tf.Variable(tf.truncated_normal([2, 2, 8]), name="weights")
  init = tf.global_variables_initializer()
  sess.run(init)
  print(sess.run(weights))
  print(weights)
  assign_weights = weights.assign(tf.truncated_normal([2, 2, 8]))
  sess.run(assign_weights)
  double_weights = weights.assign(weights * 2)
  sess.run(double_weights)
  sess.run(double_weights)
  sess.run(double_weights)
  sess.run(weights.assign_add(tf.ones_like(weights) * 10))
  sess.run(weights.assign_sub(tf.ones_like(weights) * 12))

  sess_a = tf.Session()
  sess_b = tf.Session()
  sess_a.run(weights.initializer)
  sess_b.run(weights.initializer)
  sess_a.run(weights)
  sess_b.run(weights)
  sess_a.run(weights.assign_add(tf.ones_like(weights) * 10))
  sess_b.run(weights.assign_sub(tf.ones_like(weights) * 2))
  sess_a.run(weights.assign_add(tf.ones_like(weights) * 100))
  sess_b.run(weights.assign_sub(tf.ones_like(weights) * 50))
  sess_a.close()
  sess_b.close()

  update_naive = tf.Variable(weights * 2)
  update = tf.Variable(weights.initialized_value() * 2)
  sess.run(tf.variables_initializer([update, update_naive]))
  print(sess.run(update_naive))
  print(sess.run(update))

  isess = tf.InteractiveSession()
  iaa = tf.constant(5.0, name="iaa")
  ibb = tf.constant(6.0, name="ibb")
  icc = tf.multiply(iaa, ibb, name="icc")
  print(icc.eval())
  isess.close()

  decay_weights = weights.assign(weights * 0.9)
  with sess.graph.control_dependencies([decay_weights]): square_weights = weights.assign(weights * weights)

  input_vector = tf.placeholder(tf.float32, shape=[3], name="input_vector")
  offsets = tf.constant([5, 5, 5], tf.float32, name="offsets")
  result = tf.add(input_vector, offsets, name="result")
  print(sess.run(result, {input_vector: [1, 2, 3]}))
  input_sets = [[0, 0, 0], [0, 1, 2], [3, 4, 5], [-5, 0, 5]]
  for value in input_sets: print(f"{value} -> {sess.run(result, {input_vector: value})}")

  op_input = tf.add(2, 3, name="op_input")
  operation = tf.multiply(op_input, 3, name="operation")
  sess.run(operation)
  replace_dict = {op_input: 15}
  sess.run(operation, feed_dict=replace_dict)

  x = tf.Variable(10, name="x")
  y = tf.Variable(20, name="y")
  z = tf.add(x, y, name="z")
  sess.run(tf.global_variables_initializer())
  for _ in range(10): sess.run(z)
  sess.run(tf.global_variables_initializer())
  for _ in range(10): sess.run(tf.add(x, y))

  summary_logdir = 'log_tensorboard'
  summary_writer = tf.summary.FileWriter(summary_logdir, sess.graph)
  summary_writer.close()
  for line in ['Now, at the command line, we can start up TensorBoard.', f'$ tensorboard --logdir={summary_logdir}']: print(line)
  for line in ['TensorBoard runs as a local web app, on port 6006.', '$ open http://localhost:6006/#graphs']: print(line)
"""
repl(code.strip(), globals(), locals())

