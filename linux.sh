#!/bin/bash
#sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
pip install trame ipywidgets
jupyter contrib nbextension install --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)"

