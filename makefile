# Makefile for proteusPy and associated programs
# Author: Eric G. Suchanek, PhD
# Last revision: 2024-12-17 19:05:22 -egs-


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

CONDA = conda

MESS = "0.98.2.dev1"
DEVNAME = ppydev
OUTFILES = sdist.out, bdist.out, docs.out tag.out

vers: .
	@echo "Version = $(VERS)"

newvers: .
	@echo "Current version number is: $(VERS)"	
	@echo "Enter new version number: "
	@read VERS; echo "__version__ = \"$$VERS\"" > proteusPy/_version.py
	@echo "New version number is: $(VERS)"
	@echo "Enter a new message: "
	@read MESS

nuke: clean devclean
	-@$(RM) $(OUTFILES)
	-@$(RM) dist/*

pkg:
	@echo "Starting installation step 1/2..."
	$(CONDA) create --name proteusPy -y python=3.12 numpy pandas matplotlib
	@echo "Step 1 done. Now activate the environment with 'conda activate proteusPy' and run 'make install'"

dev:
	@echo "Building $(DEVNAME)..."
	$(CONDA) create --name $(DEVNAME) -y python=3.12 numpy pandas matplotlib
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
.PHONY: install
install:
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
	
	pip install .[all] 
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
.PHONY: format
format:
	black proteusPy

bld:  docs sdist
	@echo "Building $(VERS)..."
	@echo "Building binary distribution..."
	@python setup.py bdist_wheel
	@echo "Building documentation..."
	@pdoc -o docs --math --logo "./logo.png" ./proteusPy
	@echo "Building done."

sdist: proteusPy/_version.py
	@echo "Building source distribution..."
	python setup.py sdist

docs: $(wildcard proteusPy/**/*.py)
	@echo "Building documentation..."
	pdoc -o docs --math --logo "./logo.png" ./proteusPy

# normally i push to PyPi via github action
.PHONY: upload
upload: sdist
	twine upload -r proteusPy dist/proteusPy-$(VERS)*

.PHONY: tag
tag:
	git tag -a $(VERS) -m $(MESS)
	@echo $(VERS) > tag.out

.PHONY: commit
commit:
	git commit -a -m $(MESS)
	git push origin

.PHONY: tests
tests: 
	pytest .
	python tests/Test_DisplaySS.py
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideClasses.py

# Assumes Docker is installed and running

docker: viewer/rcsb_viewer.py viewer/dockerfile
	docker build -t rcsb_viewer viewer/ --no-cache

# you have to setup the docker  cloud builder to user buildx
docker_hub: viewer/rcsb_viewer.py viewer/dockerfile
	docker buildx use cloud-egsuchanek-rcsbviewer
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
		-f viewer/dockerfile \
		-t docker.io/egsuchanek/rcsb_viewer:latest \
		-t docker.io/egsuchanek/rcsb_viewer:$(VERS) \
		--push

docker_github: viewer/rcsb_viewer.py viewer/dockerfile
	docker buildx use cloud-egsuchanek-rcsbviewer
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
		-f viewer/dockerfile \
		-t ghcr.io/suchanek/rcsb_viewer:latest \
		-t ghcr.io/suchanek/rcsb_viewer:$(VERS) \
		--push

.PHONEY: docker_all
docker_all: docker docker_hub docker_github

.PHONY: docker_run
docker_run:
	docker run -d  -p 5006:5006  --name rcsb_viewer --restart unless-stopped rcsb_viewer:latest

.PHONY: docker_purge
docker_purge:
	docker system prune -a

# end of file

