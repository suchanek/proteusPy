# makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 2/5/24 -egs-

CONDA = mamba
VERS = 0.92
DEVNAME = ppydev
INIT = proteusPy/__init__.py

FORCE: ;

dev:
	$(CONDA) env create --name $(DEVNAME) --file ppy.yml -y
	$(CONDA) install --name $(DEVNAME) pdoc -y

clean: FORCE
	$(CONDA) env remove --name proteusPy -y

devclean: FORCE
	$(CONDA) env remove --name $(DEVNAME) -y

pkg:
	$(CONDA) env create --name proteusPy --file ppy.yml -y

pkg2:
	$(CONDA) create --name proteusPy -y python=3.11.7
	
# activate the package before running!

install:
	pip install . && cd ../biopython && pip install .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name proteusPy --display-name "Python (proteusPy $(VERS) )"

install_dev:
	pip install -e . && cd ../biopython && pip install .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name ppy_dev --display-name "Python (ppy_dev)"

jup: FORCE
	sh jupyter.sh

# package development targets

sdist.out: FORCE
	rm dist/*
	python setup.py sdist
	touch sdist.out

bdist.out: FORCE
	python setup.py bdist
	touch bdist.out
docs.out: FORCE
	pdoc -o docs --math --logo "./logo.png" ./proteusPy
	touch docs.out

upload: sdist
	twine upload dist/*

tag.out: FORCE
	git tag -a $(VERS) -m $(VERS)
	touch tag.out

build: FORCE
	tag.out
	sdist.out
	docs.out


# end of file

