# Makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 2/17/24 -egs-

CONDA = mamba
# this MUST match the version in proteusPy/__init__.py
VERS = 0.92.2
DEVNAME = ppydev
INIT = proteusPy/__init__.py
OUTFILES = sdist.out, bdist.out, docs.out

FORCE: ;

nuke: clean devclean
	rm $(OUTFILES)
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
	python -m ipykernel install --user --name ppy_dev --display-name "Python (ppy_dev $(VERS) )"

# package development targets

sdist.out: FORCE
	rm dist/*
	python setup.py sdist
	@echo $(VERS) > sdist.out

bdist.out: FORCE
	python setup.py bdist
	@echo $(VERS) > bdist.out
docs.out: FORCE
	pdoc -o docs --math --logo "./logo.png" ./proteusPy
	@echo $(VERS) > docs.out

# normally i push to PyPi via github action
upload: sdist.out
	twine upload dist/*

tag.out: FORCE
	git tag -a $(VERS) -m $(VERS)
	@echo $(VERS) > tag.out

build: $(OUTFILES) 



# end of file

