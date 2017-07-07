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

  print("Hello, TensorFlow and NumPy!")
"""
repl(code.strip(), globals(), locals())

unused_code = """
  print("Unused code goes here.")
"""

