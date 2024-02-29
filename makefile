# Makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 2/25/24 -egs-

include proteusPy/__version__.py

VERS := $(shell cat proteusPy/__version__.py)
VERSION_FILE = proteusPy/__version__.py

CONDA = mamba

MESS = "disulfide module work, docs"

DEVNAME = ppydev
OUTFILES = sdist.out, bdist.out, docs.out

PHONY = .

nuke: clean devclean
	@rm $(OUTFILES)

pkg:
	@echo "Starting installation step 1/2..."
	$(CONDA) create --name proteusPy -y python=3.11.7
	@echo "Step 1 done. Now activate the environment with 'conda activate proteusPy' and run 'make install'"
dev:
	@echo "Building $(DEVNAME)..."
	$(CONDA) create --name $(DEVNAME) -y python=3.11.7
	@echo "Step 1 done. Now activate the environment with 'conda activate $(DEVNAME)' and run 'make install_dev'"

clean:
	@echo "Removing proteusPy environment..."
	@$(CONDA) env remove --name proteusPy -y

devclean:
	@echo "Removing $(DEVNAME) environment..."
	$(CONDA) env remove --name $(DEVNAME) -y

	
# activate the package before running!
install:
	@echo "Starting installation step 2/2 for $(VERS)..."
	pip install . && cd ../biopython && pip install .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name proteusPy --display-name "proteusPy $(VERS)"
	@echo "Installation finished!"

install_dev:
	@echo "Starting installation step 2/2 for $(VERS)..."
	pip install --quiet . && cd ../biopython && pip install --quiet .
	pip install pdoc twine
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name ppydev --display-name "ppydev $(VERS)"
	@echo "Installation finished!"

# package development targets
build_dev: sdist docs

sdist: .
	python setup.py sdist
	@echo $(VERS) > sdist.out

bdist: sdist
	@python setup.py bdist
	@echo $(VERS) > bdist.out

docs: sdist
	@pdoc -o docs --math --logo "./logo.png" ./proteusPy
	@echo $(VERS) > docs.out

# normally i push to PyPi via github action
upload: sdist
	twine upload dist/*

tag: sdist
	@git tag -a $(VERS) -m $(VERS)
	@echo $(VERS) > tag.out

commit:
	git commit -a -m $(MESS)
	git push --all origin

# run the tests
tests:
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideList.py
	python proteusPy/DisulfideClasses.py
	python proteusPy/turtle3D.py
	python programs/Test_DisplaySS.py

# end of file

