#!/usr/bin/env python

# Reference:
#   https://www.tensorflow.org/get_started/mnist/pros
#   https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py
#   https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py

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
"""
repl(code.strip(), globals(), locals())
print("")

code = """
def weight_variable(shape, stddev=0.01, name=None):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial, name=name)
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
def bias_variable(shape, value=0.1, name=None):
  initial = tf.constant(value, shape=shape)
  return tf.Variable(initial, name=name)
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
def max_pool_2x2(x, name=None):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
  x = tf.placeholder(tf.float32, [None, 784], name='raw_input')
  x_image = tf.reshape(x, [-1, 28, 28, 1], name='reshaped_input')

  W_conv1 = weight_variable([5, 5, 1, 32], name='weights_1')
  b_conv1 = bias_variable([32], name='biases_1')
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="convolution_1")
  h_pool1 = max_pool_2x2(h_conv1, name="pool_1")

  W_conv2 = weight_variable([5, 5, 32, 64], name='weights_2')
  b_conv2 = bias_variable([64], name='biases_2')
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="convolution_2")
  h_pool2 = max_pool_2x2(h_conv2, name="pool_2")

  W_fc1 = weight_variable([7*7*64, 1024], name='weights_fc')
  b_fc1 = bias_variable([1024], name='biases_fc')
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="pool_reshaped_for_fc")
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="pool_fc")

  keep_prob = tf.placeholder(tf.float32, name="dropout_probability")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="pool_fc_dropout")

  W_fc2 = weight_variable([1024, 10], name='weights_readout')
  b_fc2 = bias_variable([10], name='biases_readout')
  y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="output")
  y_ = tf.placeholder(tf.float32, [None, 10], name='expected_output')

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
repl(code.strip(), globals(), locals())
print("")

code = """
with tf.Session() as sess:
  steps = 20000
  interval = 100
  print(f"Training Steps: {steps}")
  print(f"Training Report Interval: Every {interval} Steps")
  sess.run(tf.global_variables_initializer())
  for i in range(steps):
    batch_x, batch_y = mnist.train.next_batch(100)
    if 0 == i % interval:
      train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
      print(f"Training Step {i}: {train_accuracy * 100}% Accuracy")
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
  final_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  print(f"Complete Training: {final_accuracy * 100}% Accuracy")
"""
repl_block(code.strip(), globals(), locals())

unused_code = """
    train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    train_step.run(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 0.5})
"""

