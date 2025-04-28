# Makefile for proteusPy and associated programs
# Author: Eric G. Suchanek, PhD
# Last revision: 2025-04-27 21:21:28 -egs-

VERS = $(shell python -c "exec(open('proteusPy/_version.py').read()); print(__version__)")
CONDA ?= conda
MESS = $(VERS)
DEVNAME = ppydev
PKGNAME = proteusPy
CONDA_PREFIX := $(shell conda info --base)
CURRENT_ENV := $(shell echo $(CONDA_DEFAULT_ENV))

OS_NAME := $(shell uname -s 2>/dev/null || echo Windows_NT)

# Repository location (can be overridden)
REPO_DIR ?= $(shell pwd)

ifeq ($(OS_NAME), Darwin)
    RM := rm -rf
else ifeq ($(OS_NAME), Linux)
    RM := rm -rf
else ifeq ($(OS_NAME), Windows_NT)
    RM := del /Q
else
    RM := rm -rf
endif

.PHONY: all vers newvers nuke pkg dev clean devclean install \
	install_dev jup jup_dev format sdist docs upload tag push-tag commit \
	tests docker docker_hub docker_github docker_all docker_run docker_purge \
	update_pyproject_version info conda_env bootstrap bld wheels

all: docs bld docker_all 

vers:
	@echo "Version = $(VERS)"
	@echo "Operating system = $(OS_NAME)"

newvers:
	@echo "Current version number is: $(VERS)"	
	@python -c "vers=input('Enter new version number: '); open('proteusPy/_version.py', 'w').write(f'__version__ = \"{vers}\"\\n')"
	$(eval VERS := $(shell python -c "exec(open('proteusPy/_version.py').read()); print(__version__)"))
	@echo "New version number is: $(VERS)"
	@echo "Updating version in pyproject.toml to $(VERS)"
	@sed -i '' 's/version = ".*"/version = "$(VERS)"/' pyproject.toml
	@echo "pyproject.toml version updated to $(VERS)"

update_pyproject_version: proteusPy/_version.py
	@echo "Updating version in pyproject.toml to $(VERS)"
	@sed -i '' 's/version = ".*"/version = "$(VERS)"/' pyproject.toml
	@echo "pyproject.toml version updated to $(VERS)"

nuke: clean devclean
	-@$(RM) dist/*

pkg:
	@echo "Starting installation step 1/2..."
	$(CONDA) create --name proteusPy -y python=3.12 numpy pandas
ifeq ($(OS_NAME), Linux)
	@echo "Linux detected, installing VTK..."
	$(CONDA) install -n proteusPy vtk -y
endif
	@echo "Step 1 done. Activate the environment with 'conda activate proteusPy' and run 'make install'"

install:
	@echo "Installing proteusPy..."
	$(CONDA) create --name $(PKGNAME) -y python=3.12
	$(CONDA) run -n $(PKGNAME) pip uninstall -y proteusPy

ifeq ($(OS_NAME), Linux)
	@echo "Linux detected, installing VTK..."
	$(CONDA) install -n $(PKGNAME) vtk -y
	$(CONDA) run -v -n $(PKGNAME) pip install dist/*.whl
else
	$(CONDA) run -v -n $(PKGNAME) pip install dist/proteuspy-$(VERS)-py3-none-any.whl
endif
	$(CONDA) run -n $(PKGNAME) python -m ipykernel install --user --name $(PKGNAME) --display-name "$(PKGNAME) ($(VERS))"
	
	@echo "proteusPy installation finished!"
	@echo "Remember to activate the environment with 'conda activate $(PKG_NAME) and run 'make bootstrap to download and build the Disulfide Databases.'"

bootstrap:
	@if [ "$(CURRENT_ENV)" != "$(PKGNAME)" ] && [ "$(CURRENT_ENV)" != "$(DEVNAME)" ]; then \
		echo "Error: Please activate either the $(PKGNAME) or $(DEVNAME) environment before running this target."; \
		exit 1; \
	fi
	@echo "Downloading and building the Disulfide Databases into $(CURRENT_ENV). This will take some time..."
	proteusPy.bootstrapper -v

dev:
	@echo "Building development environment $(DEVNAME)..."
	$(CONDA) create --name $(DEVNAME) -y python=3.12
	$(CONDA) run -n $(DEVNAME) pip install build pytest twine pdoc black
	-$(CONDA) run -n $(DEVNAME) pip uninstall -y proteusPy

ifeq ($(OS_NAME), Linux)
	@echo "Linux detected, installing VTK..."
	$(CONDA) install -n $(DEVNAME) vtk -y -q
	$(CONDA) run -v -n $(DEVNAME) pip install dist/*.whl
else
	$(CONDA) run -v -n $(DEVNAME) pip install dist/proteuspy-$(VERS)-py3-none-any.whl[all]
endif
	$(CONDA) run -n $(DEVNAME) python -m ipykernel install --user --name $(DEVNAME) --display-name "$(DEVNAME) ($(VERS))"
	@echo "Development environment installation finished. Remember to activate the environment with 'conda activate $(DEVNAME)'"
	@echo "and run 'make bootstrap to download and build the Disulfide Databases.'"

clean devclean:
	@echo "Removing $(if $(filter $@,clean),$(PKGNAME),$(DEVNAME)) environment..."
	-@jupyter kernelspec uninstall $(if $(filter $@,clean),$(PKGNAME),$(DEVNAME)) -y
	-@$(CONDA) env remove --name $(if $(filter $@,clean),$(PKGNAME),$(DEVNAME)) -y


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

bld: wheels
	@echo "Build complete."


wheels: proteusPy/_version.py
	@echo "Building wheels..."
	-@$(RM) dist/*
	python -m build --sdist --wheel .
	@echo "Wheels built successfully."

docs: $(wildcard proteusPy/**/*.py)
	@echo "Generating documentation..."
	pdoc -o docs --math --logo "./logo.png" ./proteusPy

upload: wheels
	twine upload -r proteusPy dist/proteusPy-$(VERS)*

tag:
	git tag -a $(VERS) -m $(MESS)

push-tag:
	git push origin $(VERS)

commit:
	git commit -a -m $(MESS)
	git push origin

tests:
ifeq ($(OS_NAME), Linux)
	@echo "Running tests on Linux from outside repository..."
	@mkdir -p /tmp/proteusPy_test_run
	@cd /tmp/proteusPy_test_run && python -m pytest $(REPO_DIR)/tests
	@python $(REPO_DIR)/tests/Test_DisplaySS.py
	@python $(REPO_DIR)/proteusPy/DisulfideClasses.py
	@rm -rf /tmp/proteusPy_test_run
else
	pytest .
	python tests/test_DisplaySS.py
	python proteusPy/DisulfideClasses.py
endif

docker: viewer/rcsb_viewer.py viewer/dockerfile
	docker build -t rcsb_viewer viewer/ --no-cache

docker_hub: viewer/rcsb_viewer.py viewer/dockerfile viewer/data/PDB_SS_ALL_LOADER.pkl
	docker buildx use cloud-egsuchanek-rcsbviewer
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
		-f viewer/dockerfile \
		-t docker.io/egsuchanek/rcsb_viewer:latest \
		-t docker.io/egsuchanek/rcsb_viewer:$(VERS) \
		--push --no-cache

docker_github: viewer/rcsb_viewer.py viewer/dockerfile viewer/data/PDB_SS_ALL_LOADER.pkl
	docker buildx use cloud-egsuchanek-rcsbviewer
	docker buildx build viewer/ --platform linux/arm64,linux/amd64 \
		-f viewer/dockerfile \
		-t ghcr.io/suchanek/rcsb_viewer:latest \
		-t ghcr.io/suchanek/rcsb_viewer:$(VERS) \
		--push --no-cache

docker_all: docker docker_hub docker_github

docker_run:
	docker run -d -p 5006:5006 --name rcsb_viewer --restart unless-stopped egsuchanek/rcsb_viewer:latest

docker_purge:
	docker system prune -a

info:
	@echo "Available targets in this Makefile:"
	@grep -E '^[a-zA-Z0-9_-]+:' makefile | sed 's/:.*//' | sort | uniq


# End of file
