#!/bin/sh
git-lfs track "*.csv" "*.pkl" "*.mp4"
mamba env create --file proteusPy.yml
conda activate proteusPy
pip install .
jupyter nbextension enable --py --sys-prefix widgetsnbextension
python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)"
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
