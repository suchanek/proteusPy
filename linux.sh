#!/bin/bash
#git-lfs track "*.csv" "*.pkl" "*.mp4"
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
mamba env create --file proteusPy.yml && \\ 
conda activate proteusPy && pip install . && \\
jupyter contrib nbextension install --sys-prefix && \\
jupyter nbextension enable --py --sys-prefix widgetsnbextension  && \\
python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)"

