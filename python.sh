#!/usr/bin/env sh

# Use framework version of python instead of pipenv (virtualenv) copy.
# -(which)-> python3 -(readlink)-> python3.X -(readlink)-> framework python3.X
PYTHON=$(readlink $(readlink $(which python3)))
export PYTHONHOME=$(python3 -m pipenv --venv)
exec $PYTHON $1

