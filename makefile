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

# The large .pkl files travel as assets on a dedicated data release rather than
# in git: release-asset downloads are unmetered, and a single clone of the
# LFS-tracked database used to exhaust the whole monthly LFS bandwidth quota.
# The tag is deliberately not a version tag -- the database is rebuilt only when
# the extractor reruns, so a patch release need not re-upload ~950 MB. Keep
# DATA_TAG in step with DATA_RELEASE_TAG in proteusPy/ProteusGlobals.py.
DATA_TAG ?= data-v1.0
DATA_ASSETS = proteusPy/data/PDB_all_ss.pkl \
              proteusPy/data/PDB_SS_ALL_LOADER.pkl \
              proteusPy/data/PDB_SS_SUBSET_LOADER.pkl

# Small enough to stay in git as ordinary blobs, and needed by the wheel.
# PDB_SS_SUBSET_LOADER.pkl does NOT belong here despite its name: at ~14 MB it
# is a DATA_ASSETS release asset like its two siblings (pyproject.toml's wheel
# `exclude` list agrees), and committing it trips check-added-large-files
# (--maxkb=1000).
TRACKED_PKL = proteusPy/data/SS_consensus_class_oct.pkl \
              proteusPy/data/SS_consensus_class_32.pkl \
              proteusPy/data/binary_class_metrics.pkl \
              proteusPy/data/octant_class_metrics.pkl \
              data/SS_consensus_class_oct.pkl \
              data/SS_consensus_class_32.pkl \
              data/binary_class_metrics.pkl \
              data/octant_class_metrics.pkl

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
	update_pyproject_version info conda_env bootstrap bld wheels \
	data-assets data-checksums data-restore

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
	pdoc -o docs --math --logo "./logo.png" ./proteusPy '!proteusPy.rcsb_viewer'

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

# Fetch the full prebuilt loader for the viewer image. It is a release asset, so
# the docker targets below pull it on demand instead of finding it in the tree.
viewer/data/PDB_SS_ALL_LOADER.pkl:
	@mkdir -p viewer/data
	python -c "from proteusPy.data_fetch import fetch_data_file; \
		fetch_data_file('PDB_SS_ALL_LOADER.pkl', destdir='viewer/data', verbose=True)"

data-assets:
	@for f in $(DATA_ASSETS); do \
		test -s "$$f" || { echo "missing $$f -- build it with 'make bootstrap' first"; exit 1; }; \
	done
	@gh release view $(DATA_TAG) >/dev/null 2>&1 || \
		gh release create $(DATA_TAG) --title "proteusPy data $(DATA_TAG)" \
			--notes "Disulfide database and prebuilt loaders for proteusPy. Fetched automatically by proteusPy.data_fetch."
	gh release upload $(DATA_TAG) $(DATA_ASSETS) --clobber
	@echo "Uploaded. Now run 'make data-checksums' and paste the result into proteusPy/ProteusGlobals.py."

data-checksums:
	@echo "DATA_RELEASE_SHA256 = {"
	@for f in $(DATA_ASSETS); do \
		test -s "$$f" || { echo "missing $$f"; exit 1; }; \
		printf '    "%s": "%s",\n' "$$(basename $$f)" "$$(shasum -a 256 $$f | cut -d" " -f1)"; \
	done
	@echo "}"

# One-time step completing the move off git-lfs: re-add the small .pkl files as
# ordinary blobs. Their lfs objects are gone from the server, so the only copies
# are local -- this refuses to commit a leftover pointer in their place.
data-restore:
	@for f in $(TRACKED_PKL); do \
		test -s "$$f" || { echo "missing $$f -- build it with 'make bootstrap' first"; exit 1; }; \
		head -c 40 "$$f" | grep -q "^version https://git-lfs" && \
			{ echo "$$f is still an lfs pointer, not the real file"; exit 1; }; \
		git add -- "$$f"; \
		echo "staged $$f"; \
	done
	@echo "Review with 'git diff --cached --stat', then commit."

info:
	@echo "Available targets in this Makefile:"
	@grep -E '^[a-zA-Z0-9_-]+:' makefile | sed 's/:.*//' | sort | uniq


# End of file
