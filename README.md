# Summary

**proteusPy** is a Python package specializing in the modeling and analysis of proteins of known structure with an emphasis on Disulfide Bonds. This package reprises my molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023), and relies on the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class. The turtle implements the functions ``Move``, ``Roll``, ``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space. The [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class implements methods to analyze the protein structure stabilizing element known as a *Disulfide Bond*. This class and its underlying methods are being used to perform a structural analysis of over 35,800 disulfide-bond containing proteins in the RCSB protein data bank.

# General Capabilities

1) Display Disulfides contained in the RCSB interactively in multiple display styles
2) Calculate geometric and energetic properties about these disulfides
3) Create structural classes by characterizing the disulfide torsional angles into *n* classes
4) Overlap disulfides onto a common frame of reference for display
5) Build protein backbones from a backbone dihedral angle template

*See https://suchanek.github.io/proteusPy/proteusPy.html for the API documentation with examples*

# Installation 

It's simplest to clone the repo via github since it contains all of the notebooks, data and test programs. Installation includes installing my fork of Biopython fork. This is required to rebuild the database. I highly recommend using Miniforge, especially in MacOS.

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
  $ git-lfs track "*.csv" "*.mp4"
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
  $ git-lfs track "*.csv" "*.mp4"
  $ make pkg
  $ mamba activate proteusPy
  $ make install
  ```

## General Usage

Once the package is installed one can use the existing notebooks for analysis of the RCSB Disulfide database. The ``notebooks`` directory contains all of my Jupyter notebooks and is a good place to start. The ``Analysis_2q7q.ipynb`` notebook provides an example of visualizing the lowest energy Disulfide contained in the database, and searching for nearest neighbors on the basis of conformational similarity. The ``programs`` subdirectory contains the primary programs for downloading the RCSB disulfide-containing structure files, (``DisulfideDownloader.py``), extracting the disulfides and creating the database loaders (``DisulfideExtractor.py``) and cluster analysis, (``DisulfideClass_Analysis.py``).

The first time one loads the database via ``Load_PDB_SS()`` the system will attempt to download the full and subset database from the Github repository. If this fails the system will attempt to rebuild the database from the repo's ``data`` subdirectory (not the package's). If you've downloaded from github this will work correctly. If you've installed from pyPi via ``pip`` it will fail.

## The Future

I am continuing to explore the initial disulfide structural classes described by Hogg *et al.* using the sextant class approach. This offers much higher class resolution and reveals subgroups within the broad class. I'd also like to explore the catalytic and allosteric classes in more detail to look for common structural elements.

## Publications

- <https://doi.org/10.1021/bi00368a023>
- <https://doi.org/10.1021/bi00368a024>
- <https://doi.org/10.1016/0092-8674(92)90140-8>
- <http://dx.doi.org/10.2174/092986708783330566>

*NB:* This distribution is being developed slowly. proteusPy relies on my fork of the ``Bio`` Python package to download and build the database. This will be installed by default if one follows the installation instructions. -egs-

Eric G. Suchanek, PhD., <mailto:suchanek@mac.com>
