#!/bin/bash ~/.bash_profile
conda activate proteus
pdoc proteusPy —math -o docs —logo "./logo.png"
python setup.py sdist bdist-wheel
