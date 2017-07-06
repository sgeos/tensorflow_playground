# Tensorflow Playground

Tensorflow studies captured in file form.

The files in this project contain a function that emulates REPL output.
This way the files can be run stand alone but also out the evaluation of each line.
There is probably a better way to achieve the same effect.

## Usage Example

```sh
# terminal a
rm -rf log*
./tensorflow_tutorial.py
tensorboard --logdir=log_tutorial

# terminal b or browser
open http://localhost:6006/#graphs
open http://localhost:6006/#events
```

