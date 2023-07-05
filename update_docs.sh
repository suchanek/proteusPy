#!/bin/bash
conda activate proteus
pdoc proteusPy —math -o docs —logo "./logo.png"
python setup.py sdist
