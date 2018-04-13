#!/bin/sh

#reset
. ${HOME}/tensorflow_virtualenv/bin/activate
tensorboard --logdir=log_mnist_with_summaries
#echo 'reset ; . ~/tensorflow_virtualenv/bin/activate ; tensorboard --logdir=log_mnist_with_summaries'

