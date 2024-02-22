# Summary

**proteusPy** is a Python package specializing in the modeling and analysis of proteins of known structure with an emphasis on Disulfide bonds. This package reprises my molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023), a structure-based program developed as part of my graduate thesis. The package relies on the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class to create and manipulate local coordinate systems. It does this by implementing the functions ``Move``, ``Roll``, ``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space.  The initial implementation focuses on the [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class. The class implements methods to analyze the protein structure stabilizing element known as a *Disulfide Bond*. This class and its underlying methods are being used to perform a structural analysis of over 35,800 disulfide-bond containing proteins in the RCSB protein data bank (https://www.rcsb.org).

# General Capabilities
- Interactively display disulfides contained in the RCSB in a variety of display styles
- Calculate geometric and energetic properties about these disulfides
- Create binary and sextant structural classes by characterizing the disulfide torsional angles into *n* classes
- Build idealized disulfide bonds from dihedral angle input
- Find disulfide neighbors based on dihedral angle input
- Overlap disulfides onto a common frame of reference for display
- Build protein backbones from backbone phi, psi dihedral angle templates
- More in development

*See https://suchanek.github.io/proteusPy/proteusPy.html for the API documentation with examples*.

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

# Testing
I currently have docstring testing for the modules in place. To run them ``cd`` into the repository and run:
```console
$ make tests
```
The modules will run their docstring tests and disulfide visualization windows will open. Simply close them. If all goes normally there will be no errors.


# General Usage

Once the package is installed one can use the existing notebooks for analysis of the RCSB Disulfide database. 

The [notebooks](https://github.com/suchanek/proteusPy/blob/master/notebooks/) directory contains all of my Jupyter notebooks and is a good place to start: 
- [Analysis_2q7q.ipynb](https://github.com/suchanek/proteusPy/blob/master/notebooks/Analysis_2q7q.ipynb) provides an example of visualizing the lowest energy Disulfide contained in the database and searching for nearest neighbors on the basis of conformational similarity. 

The [programs](https://github.com/suchanek/proteusPy/tree/master/programs) subdirectory contains the primary programs for downloading the RCSB disulfide-containing structure files:
* [DisulfideDownloader.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideDownloader.py): Downloads the raw RCSB structure files.
* [DisulfideExtractor.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideExtractor.py): Extracts the disulfides and creating the database loaders
* [DisulfideClass_Analysis.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideExtractor.py): Performs binary or sextant analysis on the disulfide database.

The first time one loads the database via [Load_PDB_SS()](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html#Load_PDB_SS) the system will attempt to download the full and subset database from Google Drive. If this fails the system will attempt to rebuild the database from the repo's **data** subdirectory (not the package's). If you've downloaded from github this will work correctly. If you've installed from pyPi via **pip** it will fail.


## Quickstart

After installation is complete launch jupyter lab:

```console
$ jupyter lab 
```
and open ``notebooks/Analysis_2q7q.ipynb``. This notebook looks at the disulfide bond with the lowest energy in the entire database. There are several other notebooks in this directory that illustrate using the program. Some of these reflect active development work so may not be 'fully baked'.

## Visualizing Disulfides with pyVista
PyVista is an excellent 3D visualization framework and I've used it for the Disulfide visualization engine. It uses the VTK library on the back end and provides high-level access to 3d rendering. The menu strip provided in the Disulfide visualization windows allows the user to turn borders, rulers, bounding boxes on and off and reset the orientations. Please try them out! There is also a button for *local* vs *server* rendering. *Local* rendering is usually much smoother. To manipulate:
- Click and drag your mouse to rotate
- Use the mouse wheel to zoom (3 finger zoom on trackpad)

## Performance
- Manipulating and searching through long lists of disulfides can take time. I've added progress bars for many of these operations. 
- Rendering many disulfides in **pyvista** can also take time to load and may be slow to dispay in real time, depending on your hardware. I added optimization to reduce cylinder complexity as a function of total cylinders rendered, but it can still be less than perfect. The faster your GPU the better! 

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
- <http://dx.doi.org/10.1111/j.1538-7836.2010.03894.x>

