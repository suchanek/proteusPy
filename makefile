# Makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 2/20/24 -egs-

CONDA = mamba

# this MUST match the version in proteusPy/__init__.py
VERS = 0.92.6
MESS = "JOSS work, docs, added fxns to Disulfide.py for modeling"

DEVNAME = ppydev
OUTFILES = sdist.out, bdist.out, docs.out
nuke: clean devclean
	@rm $(OUTFILES)

dev:
	@echo "Building... $(DEVNAME)"
	@$(CONDA) env create --name $(DEVNAME) --file ppy.yml -y -q
	@$(CONDA) install --name $(DEVNAME) pdoc -y -q
	@echo "Step 1 done. Now activate the environment with 'conda activate $(DEVNAME)' and run 'make install'"

clean:
	@echo "Removing proteusPy environment..."
	@$(CONDA) env remove --name proteusPy -y

devclean:
	@echo "Removing $(DEVNAME) environment..."
	$(CONDA) env remove --name $(DEVNAME) -y

pkg:
	@echo "Starting installation step 1/2..."
	@$(CONDA) env create --name proteusPy --file ppy.yml -y -q
	@echo "Step 1 done. Now activate the environment with 'conda activate proteusPy' and run 'make install'"

pkg2:
	$(CONDA) create --name ppy2 -y python=3.11.7
	
# activate the package before running!

install:
	@echo "Starting installation step 2/2..."
	@pip install --quiet . && cd ../biopython && pip install --quiet .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name proteusPy --display-name "proteusPy $(VERS)"
	

install_dev:
	@pip install -e . && cd ../biopython && pip install .
	@jupyter contrib nbextension install --sys-prefix
	@jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name ppy_dev --display-name "ppy_dev $(VERS)"

# package development targets
build_dev: sdist docs

sdist: proteusPy/__init__.py
	@python setup.py sdist
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

# run the docstring tests
tests:
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideList.py
	python proteusPy/DisulfideClasses.py
	python proteusPy/turtle3D.py

# end of file

