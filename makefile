# Makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 3/16/24 -egs-

# assumes file VERSION contains only the version number
ifeq ($(OS),Windows_NT) 
    VERS := $(shell python -c "exec(open('proteusPy/version.py').read()); print(__version__)")
else 
	VERS = $(shell python get_version.py)
endif

CONDA = mamba

MESS = $(VERS)
DEVNAME = ppydev
OUTFILES = sdist.out, bdist.out, docs.out tag.out

PHONY = .

vers: .
	@echo "Version = $(VERS)"

newvers: .
	@echo "Current version number is: $(VERS)"	
	@echo "Enter new version number: "
	@read VERS; echo "__version__ = \"$$VERS\"" > proteusPy/version.py
	@echo "New version number is: $(VERS)"
	@echo "Enter a new message: "
	@read MESS

nuke: clean devclean
	ifeq ($(OS),Windows_NT) 
		@del $(OUTFILES)
		@del dist/*
	else 
		@rm $(OUTFILES)
		@rm dist/*
	endif

pkg:
	@echo "Starting installation step 1/2..."
	$(CONDA) create --name proteusPy -y python=3.11.7
	@echo "Step 1 done. Now activate the environment with 'conda activate proteusPy' and run 'make install'"

dev:
	@echo "Building $(DEVNAME)..."
	$(CONDA) create --name $(DEVNAME) -y python=3.11.7
	@echo "Step 1 done. Now activate the environment with 'conda activate $(DEVNAME)' and run 'make install_dev'"

clean: .
	@echo "Removing proteusPy environment..."
	-@jupyter kernelspec uninstall proteuspy -y
	-@$(CONDA) env remove --name proteusPy -y

devclean: .
	@echo "Removing $(DEVNAME) environment..."
	-@jupyter kernelspec uninstall ppydev -y
	-@$(CONDA) env remove --name $(DEVNAME) -y

	
# activate the package before running!
install: 
	@echo "Starting installation step 2/2 for $(VERS)..."
	@echo "Installing VTK..."
	@echo "Installing proteusPy..."
	pip install dist/proteusPy-$(VERS).tar.gz
	@echo "Installing Biopython..."
	@pip install git+https://github.com/suchanek/biopython.git@egs_ssbond_240305#egg=biopython
	@echo "Installing jupyter..."
	@jupyter contrib nbextension install --sys-prefix
	@jupyter nbextension enable --py --sys-prefix widgetsnbextension
	@python -m ipykernel install --user --name proteusPy --display-name "proteusPy ($(VERS))"
	@echo "Installation finished!"

install_dev:
	@echo "Starting installation step 2/2 for $(VERS)..."
	$(CONDA) install -y vtk
	#pip install dist/proteusPy-$(VERS)-py3-none-any.whl
	pip install -e .
	pip install git+https://github.com/suchanek/biopython.git@egs_ssbond_240305#egg=biopython
	pip install pdoc twine black pytest build

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

# package development targets

format: .
	black proteusPy

bld:  format  docs sdist

sdist: .
	python -m build
	@echo $(VERS) > sdist.out

docs: .
	pdoc -o docs --math --logo "./logo.png" ./proteusPy
	@echo $(VERS) > docs.out

# normally i push to PyPi via github action
upload: sdist
	twine upload -r proteusPy dist/proteusPy-$(VERS)*

tag: .
	git tag -a $(VERS) -m $(MESS)
	@echo $(VERS) > tag.out

commit:
	git commit -a -m $(MESS)
	git push --all origin

# run the tests
tests: .
	pytest .
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideList.py
	python proteusPy/DisulfideClasses.py

# end of file

