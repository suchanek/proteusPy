#!/bin/bash
cd ~/repos/proteusPy
conda activate proteusPy && \\ 
pdoc -o docs --math --logo "./logo.png" ./proteusPy && \\
python -m build

