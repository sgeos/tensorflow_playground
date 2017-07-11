#!/usr/bin/env python

# Reference:
#   https://www.tensorflow.org/get_started/mnist/mechanics
#   https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist.py
#   https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/fully_connected_feed.py

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
  import os.path
  import sys
  import time
  from six.moves import xrange

  import tensorflow as tf
  from tensorflow.examples.tutorials.mnist import input_data
  from tensorflow.examples.tutorials.mnist import mnist

  FLAGS = {}
  FLAGS['learning_rate'] = 0.01
  FLAGS['max_steps'] = 1000000
  FLAGS['hidden1'] = 128
  FLAGS['hidden2'] = 32
  FLAGS['batch_size'] = 1000
  FLAGS['print_size'] = 1000
  FLAGS['print_dots'] = 50
  FLAGS['snapshot_size'] = 10000
  FLAGS['input_data_dir'] = 'MNIST/input_data'
  FLAGS['log_dir'] = 'MNIST/logs/tensorflow_mechanics_101'
  FLAGS['fake_data'] = False
"""
repl(code.strip(), globals(), locals())
print("")

code = """
def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS['batch_size'], FLAGS['fake_data'])
  feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
  return feed_dict
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
  true_count = 0
  steps_per_epoch = data_set.num_examples // FLAGS['batch_size']
  num_examples = steps_per_epoch * FLAGS['batch_size']
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))
"""
repl_block(code.strip(), globals(), locals())
print("")

code = """
  if tf.gfile.Exists(FLAGS['log_dir']): tf.gfile.DeleteRecursively(FLAGS['log_dir'])
  tf.gfile.MakeDirs(FLAGS['log_dir'])

  data_sets = input_data.read_data_sets(FLAGS['input_data_dir'], FLAGS['fake_data'])
"""
repl(code.strip(), globals(), locals())
print("")

code = """
with tf.Graph().as_default():
  images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS['batch_size'], mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS['batch_size']))
  logits = mnist.inference(images_placeholder, FLAGS['hidden1'], FLAGS['hidden2'])
  loss = mnist.loss(logits, labels_placeholder)
  train_op = mnist.training(loss, FLAGS['learning_rate'])
  eval_correct = mnist.evaluation(logits, labels_placeholder)
  summary = tf.summary.merge_all()
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  sess = tf.Session()
  summary_writer = tf.summary.FileWriter(FLAGS['log_dir'], sess.graph)
  sess.run(init)

  for _ in range(FLAGS['print_dots'] - 1): print('.', end='')
  for step in xrange(FLAGS['max_steps']):
    start_time = time.time()
    feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    duration = time.time() - start_time

    if 0 == (step + 1) % FLAGS['snapshot_size'] or (step + 1) == FLAGS['max_steps']:
      print(' ', end='')
      checkpoint_file = os.path.join(FLAGS['log_dir'], 'model.ckpt')
      saver.save(sess, checkpoint_file, global_step=step)
      print('Training Data Eval:')
      do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
      print('Validation Data Eval:')
      do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
      print('Test Data Eval:')
      do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)
      for _ in range(FLAGS['print_dots'] - 1): print('.', end='')

    if 0 == step % FLAGS['print_size']:
      print(' Step %d: loss = %.2f (%.3f sec) ' % (step, loss_value, duration))
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()
    elif 0 == step % (FLAGS['print_size'] / FLAGS['print_dots']):
      print('.', end='')
      sys.stdout.flush()

  summary_writer.close()
  print(' Training complete.')
  print('')
  for line in ['Now, at the command line, we can start up TensorBoard.', f"$ tensorboard --logdir={FLAGS['log_dir']}"]: print(line)
  for line in ['TensorBoard runs as a local web app, on port 6006.', '$ open http://localhost:6006/#graphs', '$ open http://localhost:6006/#events']: print(line)
"""
repl_block(code.strip(), globals(), locals())

unused_code = """
  saver.restore(sess, FLAGS.train_dir)
"""

