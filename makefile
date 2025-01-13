# Makefile for proteusPy and associated programs
# Author: Eric G. Suchanek, PhD
# Last revision: 2024-12-17 19:05:22 -egs-

VERS := $(shell python -c "exec(open('proteusPy/_version.py').read()); print(__version__)")
RM := rm
CONDA ?= conda
MESS = "v0.98.31"
DEVNAME = ppydev

.PHONY: all vers newvers nuke pkg dev clean devclean install install_dev jup jup_dev format bld sdist docs upload tag push-tag commit tests docker docker_hub docker_github docker_all docker_run docker_purge

all: clean sdist docs bld

vers:
	@echo "Version = $(VERS)"

newvers:
	@echo "Current version number is: $(VERS)"	
	@python -c "vers=input('Enter new version number: '); open('proteusPy/_version.py', 'w').write(f'__version__ = \"{vers}\"\\n')"
	@echo "Version number updated."

nuke: clean devclean
	-@$(RM) dist/*

pkg:
	@echo "Starting installation step 1/2..."
	$(CONDA) create --name proteusPy -y python=3.12 numpy pandas
	@echo "Step 1 done. Activate the environment with 'conda activate proteusPy' and run 'make install'"

dev:
	@echo "Building development environment $(DEVNAME)..."
	$(CONDA) create --name $(DEVNAME) -y python=3.12 numpy pandas
	@echo "Step 1 done. Activate the environment with 'conda activate $(DEVNAME)' and run 'make install_dev'"

clean devclean:
	@echo "Removing $(if $(filter $@,clean),proteusPy,$(DEVNAME)) environment..."
	-@jupyter kernelspec uninstall $(if $(filter $@,clean),proteusPy,$(DEVNAME)) -y
	-@$(CONDA) env remove --name $(if $(filter $@,clean),proteusPy,$(DEVNAME)) -y

install:
	@echo "Starting installation step 2/2 for $(VERS)..."
	pip install .
	python -m ipykernel install --user --name proteusPy --display-name "proteusPy ($(VERS))"
	@echo "Installation finished!"

install_dev: sdist
	@echo "Starting installation step 2/2 for $(VERS)..."
	pip install .[all]
	pip install pdoc twine black pytest build -q
	python -m ipykernel install --user --name $(DEVNAME) --display-name "$(DEVNAME) ($(VERS))"
	@echo "Development environment installation finished!"

define jupyter-setup
	jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	python -m ipykernel install --user --name $(1) --display-name "$(1) ($(VERS))"
endef

jup:
	$(call jupyter-setup,proteusPy)

jup_dev:
	$(call jupyter-setup,ppydev)

format:
	black proteusPy

bld: docs sdist
	@echo "Building $(VERS)..."
	python setup.py bdist_wheel
	pdoc -o docs --math --logo "./logo.png" ./proteusPy
	@echo "Build complete."

sdist: proteusPy/_version.py
	@echo "Building source distribution..."
	python setup.py sdist

docs: $(wildcard proteusPy/**/*.py)
	@echo "Generating documentation..."
	pdoc -o docs --math --logo "./logo.png" ./proteusPy

upload: sdist
	twine upload -r proteusPy dist/proteusPy-$(VERS)*

tag:
	git tag -a $(VERS) -m $(MESS)
	@echo $(VERS) > tag.out

push-tag:
	git push origin $(VERS)

commit:
	git commit -a -m $(MESS)
	git push origin

tests: 
	pytest .
	python tests/Test_DisplaySS.py
	python proteusPy/Disulfide.py
	python proteusPy/DisulfideLoader.py
	python proteusPy/DisulfideClasses.py

docker: viewer/rcsb_viewer.py viewer/dockerfile
	docker build -t rcsb_viewer viewer/ --no-cache

docker_hub: viewer/rcsb_viewer.py viewer/dockerfile
	docker buildx use cloud-egsuchanek-rcsbviewer
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
		-f viewer/dockerfile \
		-t docker.io/egsuchanek/rcsb_viewer:latest \
		-t docker.io/egsuchanek/rcsb_viewer:$(VERS) \
		--push --no-cache

docker_github: viewer/rcsb_viewer.py viewer/dockerfile
	docker buildx use cloud-egsuchanek-rcsbviewer
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
		-f viewer/dockerfile \
		-t ghcr.io/suchanek/rcsb_viewer:latest \
		-t ghcr.io/suchanek/rcsb_viewer:$(VERS) \
		--push --no-cache

docker_all: docker docker_hub docker_github

docker_run:
	docker run -d -p 5006:5006 --name rcsb_viewer --restart unless-stopped rcsb_viewer:latest

docker_purge:
	docker system prune -a -y

# End of file
