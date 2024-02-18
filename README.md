# Summary

**proteusPy** is a Python package specializing in the modeling and analysis of proteins of known structure with an emphasis on Disulfide bonds. This package reprises my molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023), a structure-based program developed as part of my graduate thesis. The package relies on the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class to create and manipulate local coordinate systems. The turtle implements the functions ``Move``, ``Roll``, ``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space. This initial implementation focuses on the [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class. The class implements methods to analyze the protein structure stabilizing element known as a *Disulfide Bond*. This class and its underlying methods are being used to perform a structural analysis of over 35,800 disulfide-bond containing proteins in the RCSB protein data bank (https://www.rcsb.org).

# General Capabilities
- Interactively display disulfides contained in the RCSB in a variety of display styles
- Calculate geometric and energetic properties about these disulfides
- Create binary and sextant structural classes by characterizing the disulfide torsional angles into *n* classes
- Build idealized disulfide bonds from dihedral angle input
- Find disulfide neighbors based on dihedral angle input
- Overlap disulfides onto a common frame of reference for display
- Build protein backbones from backbone phi, psi dihedral angle templates
- More in development

*See https://suchanek.github.io/proteusPy/proteusPy.html for the API documentation with examples*

# Requirements

1. PC running MacOS, Linux, Windows with git, git-lfs and C compiler installed.
2. 8 GB RAM
3. 3 GB disk space

# Installation 

It's simplest to clone the repo via GitHub since it contains all of the notebooks, data and test programs. Installation includes installing my Biopython fork which is required to rebuild the database (this is not needed generally). I highly recommend using Miniforge since it includes mamba. The installation instructions below assume a clean install with no package manager or compiler installed.

## MacOS/Linux
- Install Miniforge: <https://github.com/conda-forge/miniforge> (existing Anaconda installations are fine but please install mamba)
- Install git-lfs:
  - <https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage>
- Install `make` on your system.
- From a shell prompt while sitting in your repo dir:
  ```console
  $ git clone https://github.com/suchanek/proteusPy.git
  $ git clone https://github.com/suchanek/biopython.git
  $ cd proteusPy
  $ git-lfs track "*.csv" "*.mp4" "*.pkl"
  $ make pkg
  $ mamba activate proteusPy
  $ make install
  ```
## Windows
- Install Miniforge: <https://github.com/conda-forge/miniforge> (existing Anaconda installations are fine but please install mamba)
- Install git for Windows and configure for Bash:
  - https://git-scm.com/download/win
- Install git-lfs:
  - https://git-lfs.github.com/
- Install Visual Studio C++ development environment: https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170
- From an Anaconda Powershell prompt while sitting in your repo dir:
  ```console
  $ git clone https://github.com/suchanek/proteusPy.git
  $ git clone https://github.com/suchanek/biopython.git
  $ cd proteusPy
  $ git-lfs track "*.csv" "*.mp4" "*.pkl"
  $ make pkg
  $ mamba activate proteusPy
  $ make install
  ```

## General Usage

Once the package is installed one can use the existing notebooks for analysis of the RCSB Disulfide database. The ``notebooks`` directory contains all of my Jupyter notebooks and is a good place to start. The ``Analysis_2q7q.ipynb`` notebook provides an example of visualizing the lowest energy Disulfide contained in the database, and searching for nearest neighbors on the basis of conformational similarity. The ``programs`` subdirectory contains the primary programs for downloading the RCSB disulfide-containing structure files, (``DisulfideDownloader.py``), extracting the disulfides and creating the database loaders (``DisulfideExtractor.py``) and cluster analysis, (``DisulfideClass_Analysis.py``).

The first time one loads the database via ``Load_PDB_SS()`` the system will attempt to download the full and subset database from the GitHub repository. If this fails the system will attempt to rebuild the database from the repo's ``data`` subdirectory (not the package's). If you've downloaded from GitHub this will work correctly. If you've installed from pyPi via ``pip`` it will fail.

## Quickstart

After installation is complete launch jupyter lab:

```console
$ jupyter lab 
```
and open ``notebooks/Analysis_2q7q.ipynb``. This notebook looks at the disulfide bond with the lowest energy in the entire database. There are several other notebooks in this directory that illustrate using the program. Some of these reflect active development work so may not be 'fully baked'.

## The Future

I am continuing to explore the initial disulfide structural classes described by Hogg *et al.* using the sextant class approach. This offers much higher class resolution and reveals subgroups within the broad class. I'd also like to explore the catalytic and allosteric classes in more detail to look for common structural elements.

## Citing proteusPy

The proteusPy package was developed by Eric G. Suchanek, PhD. If you find it useful in your research and wish to cite it please use the following BibTeX entry:

```
@software{proteusPy2024,
  author = {Eric G. Suchanek, PhD},
  title = {proteusPy: A Package for Modeling and Analyzing Proteins of Known Structure},
  year = {2024},
  publisher = {GitHub},
  version = {0.92},
  journal = {GitHub repository},
  url = {https://github.com/suchanek/proteusPy}
}
```

## Publications

- <https://doi.org/10.1021/bi00368a023>
- <https://doi.org/10.1021/bi00368a024>
- <https://doi.org/10.1016/0092-8674(92)90140-8>
- <http://dx.doi.org/10.2174/092986708783330566>
