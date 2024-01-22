#!/bin/bash
git-lfs track "*.csv" "*.pkl" "*.mp4"
mamba env create --file proteusPy.yml
conda activate proteusPy && \
sudo jupyter contrib nbextension install --sys-prefix && \\
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super && \\
jupyter nbextension enable --py --sys-prefix widgetsnbextension  && \\
python -m ipykernel install --user--name proteusPy --display-name "Python (proteusPy)"  && \\
pip install .
conda activate proteusPy

