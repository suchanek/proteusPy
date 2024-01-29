git clone https://github.com/suchanek/proteusPy
cd .\proteusPy\
mamba env create --name proteusPy --file=proteusPy.yml
conda activate proteusPy
pip install .
Jupyter contrib nbextension install --sys-prefix
python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)"
jupyter nbextension enable --py --sys-prefix widgetsnbextension
