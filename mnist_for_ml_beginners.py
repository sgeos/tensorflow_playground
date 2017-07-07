#!/usr/bin/env python

# Reference:
#   https://www.tensorflow.org/get_started/mnist/beginners
#   https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py

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

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784], name='input')
  W = tf.Variable(tf.random_normal([784, 10], 0.0, 0.1, tf.float32), name='weights')
  b = tf.Variable(tf.random_normal([10], 0.0, 0.01, tf.float32), name='biases')
  y = tf.nn.softmax(tf.matmul(x, W) + b, name='output')
  y_ = tf.placeholder(tf.float32, [None, 10], name='expected_output')

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
"""
repl(code.strip(), globals(), locals())
print("")

code = """
for _ in range(1000000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
"""
repl(code.strip(), globals(), locals())

unused_code = """
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
for _ in range(10000): sess.run(train_step, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
"""

