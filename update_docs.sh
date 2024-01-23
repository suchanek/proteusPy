#!/bin/bash
conda activate proteus
cd ~/repos/proteusPy
pdoc proteusPy —math -o docs —logo "./logo.png"
python -m build

