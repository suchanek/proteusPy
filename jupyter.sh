#!/bin/bash
# mamba install trame ipywidgets -y
jupyter contrib nbextension install --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)"

