# Makefile for proteusPy
# Author: Eric G. Suchanek, PhD
# Last revision: 8/23/24 -egs-


# assumes file VERSION contains only the version number
ifeq ($(OS),Windows_NT) 
    VERS := $(shell python -c "exec(open('proteusPy/_version.py').read()); print(__version__)")
	RM = del
else 
	VERS = $(shell python get_version.py)
	RM = rm

endif

# mamba is better than conda. Install it with 'conda install mamba -n base -c conda-forge'
# or use conda instead of mamba.

CONDA = mamba

#MESS = $(VERS)
#MESS = "proteusPy: A Python Package for Protein Structure and Disulfide Bond Modeling and Analysis"

MESS = "0.97.14 Disulfide Viewer programs functional"
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
	-@$(RM) $(OUTFILES)
	-@$(RM) dist/*

pkg:
	@echo "Starting installation step 1/2..."
	$(CONDA) create --name proteusPy -y python=3.11.7 numpy pandas matplotlib
	@echo "Step 1 done. Now activate the environment with 'conda activate proteusPy' and run 'make install'"

dev:
	@echo "Building $(DEVNAME)..."
	$(CONDA) create --name $(DEVNAME) -y python=3.11.7 numpy pandas matplotlib
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

install: sdist
	@echo "Starting installation step 2/2 for $(VERS)..."
	@echo "Installing additional..."
	$(CONDA) install vtk==9.2.6 -y

	@echo "Installing proteusPy..."
	@pip install . -q
	
	@echo "Installing jupyter..."
	@python -m ipykernel install --user --name proteusPy --display-name "proteusPy ($(VERS))"
	@echo "Installation finished!"

install_dev: sdist
	@echo "Starting installation step 2/2 for $(VERS)..."
	$(CONDA) install vtk==9.2.6 -y
	
	pip install . -q
	pip install pdoc twine black pytest build -q
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

bld:  docs sdist

sdist: proteusPy/_version.py
	python setup.py sdist

.PHONY: docs
docs: proteusPy/_version.py
	pdoc -o docs --math --logo "./logo.png" ./proteusPy

# normally i push to PyPi via github action
.PHONY: upload
upload: sdist
	twine upload -r proteusPy dist/proteusPy-$(VERS)*

tag:
	git tag -a $(VERS) -m $(MESS)
	@echo $(VERS) > tag.out

commit:
	git commit -a -m $(MESS)
	git push origin

# run the tests

.PHONY: tests
tests: 
	pytest .
	python tests/Test_DisplaySS.py
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideClasses.py

.PHONY: docker
docker: viewer/rcsb_viewer.py viewer/dockerfile
	docker build -t rcsb_viewer:latest viewer/ --no-cache

.PHONY: docker_hub
docker_hub: .
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
	 -f viewer/dockerfile \
	 -t docker.io/egsuchanek/rcsb_viewer:latest --push --no-cache


.PHONY: docker_github
docker_github: docker
	docker tag rcsb_viewer:latest ghcr.io/suchanek/rcsb_viewer:latest
	docker push ghcr.io/suchanek/rcsb_viewer:latest 

# end of file

