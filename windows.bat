@echo off
REM git clone https://github.com/suchanek/proteusPy
REM cd .\proteusPy\
REM mamba env create --name proteusPy --file=proteusPy.yml -y
conda activate proteusPy & pip install . & Jupyter contrib nbextension install --sys-prefix & python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)" & jupyter nbextension enable --py --sys-prefix widgetsnbextension

