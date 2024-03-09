# Makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 3/8/24 -egs-

VERS := $(shell grep ^0 VERSION | cut -d= -f2 | tr -d \" | sed 's/^[[:space:]]*//')

PYPI_PASSWORD := $(shell echo $$PYPI_PASSWORD)
CONDA = mamba

MESS = "more cleanup"

DEVNAME = ppydev
OUTFILES = sdist.out, bdist.out, docs.out tag.out

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

clean: remove_jupyter
	@echo "Removing proteusPy environment..."
	@$(CONDA) env remove --name proteusPy -y

devclean:
	@echo "Removing $(DEVNAME) environment..."
	$(CONDA) env remove --name $(DEVNAME) -y

	
# activate the package before running!
install:
	@echo "Starting installation step 2/2 for $(VERS)..."
	pip install -U . 
	pip install git+https://github.com/suchanek/biopython.git@egs_ssbond_240305
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name proteusPy --display-name "proteusPy $(VERS)"
	@echo "Installation finished!"

install_dev:
	@echo "Starting installation step 2/2 for $(VERS)..."
	pip install -U .
	pip install git+https://github.com/suchanek/biopython.git@egs_ssbond_240305#egg=biopython
	pip install pdoc twine black
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name ppydev --display-name "ppydev ($(VERS))"
	@echo "Installation finished!"

jup: .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name proteusPy --display-name "proteusPy ($(VERS))"

jup_dev: .
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name ppydev --display-name "ppydev ($(VERS))"


remove_jupyter: 
	jupyter kernelspec uninstall proteuspy -y

remove_jupyter_dev: 
	jupyter kernelspec uninstall ppydev -y

# package development targets

format: sdist
	black proteusPy

bld: sdist docs format

sdist: .
	python -m build
	@echo $(VERS) > sdist.out

docs: .
	@pdoc -o docs --math --logo "./logo.png" ./proteusPy
	@echo $(VERS) > docs.out

# normally i push to PyPi via github action
upload: sdist
	twine upload -r proteusPy dist/proteusPy-$(VERS)*
tag: .
	@git tag -a $(VERS) -m $(MESS)
	@echo $(VERS) > tag.out

commit:
	git commit -a -m $(MESS)
	git push --all origin

# run the tests
tests: .
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideList.py
	python proteusPy/DisulfideClasses.py
	python proteusPy/turtle3D.py
	python tests/Test_DisplaySS.py

# end of file

