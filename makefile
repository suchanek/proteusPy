# makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 2/5/24 -egs-

CONDA = mamba
VERS = 0.89
INIT = proteusPy/__init__.py

FORCE: ;

dev:
	$(CONDA) env create --name ppy_dev --file ppy.yml -y
	$(CONDA) install --name ppy_dev pdoc -y

clean: FORCE
	$(CONDA) env remove --name proteusPy -y

pkg:
	$(CONDA) env create --name proteusPy --file ppy.yml -y

pkg2:
	$(CONDA) create --name proteusPy -y python=3.11.7
	
# activate the package before running!

install:
	pip install . && cd ../biopython && pip install .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy)"

install_dev:
	pip install . && cd ../biopython && pip install .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name ppy_dev --display-name "Python (ppy_dev)"


jup: FORCE
	sh jupyter.sh

# package development targets

sdist: setup.py
	rm dist/*
	python setup.py sdist

docs: FORCE
	pdoc -o docs --math --logo "./logo.png" ./proteusPy

upload: sdist
	twine upload dist/*

tag: FORCE
	git tag -a $(VERS) -m $(VERS)

build: FORCE
	tag
	sdist
	docs


# end of file

