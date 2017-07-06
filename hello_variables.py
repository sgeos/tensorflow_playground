#!/usr/bin/env python

def repl( expression, globals=None, locals=None ):
  for expression in expression.splitlines():
    expression = expression.strip()
    if expression:
      print(f">>> {expression}")
      eval(compile(expression + "\n", "<string>", "single"), globals, locals)
    else:
      print("")

code = """
  foo = []
  bar = foo
  foo.append(bar)
  (foo, bar)
  (id(foo), id(bar))

  foo == bar
  foo is bar

  3
  [1., 2., 3.]
  [[1., 2., 3.], [4., 5., 6.]]
  [[[1., 2., 3.]], [[7., 8., 9.]]]
"""
repl(code.strip(), globals(), locals())

