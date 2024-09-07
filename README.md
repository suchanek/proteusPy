
[![PyPI version](https://badge.fury.io/py/proteusPy.svg)](https://badge.fury.io/py/proteusPy)
![Testing](https://github.com/suchanek/proteusPy/actions/workflows/pytest.yml/badge.svg)
[![status](https://joss.theoj.org/papers/45de839b48a550d6ab955c5fbbc508f2/status.svg)](https://joss.theoj.org/papers/45de839b48a550d6ab955c5fbbc508f2)
[![DOI](https://zenodo.org/badge/575657091.svg)](https://doi.org/10.5281/zenodo.13241499)
[![API Docs](https://img.shields.io/badge/API%20Documentation-8A2BE2)](https://suchanek.github.io/proteusPy/proteusPy.html)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

# Summary

**proteusPy** is a Python package specializing in the modeling and analysis of proteins of known structure with an emphasis on Disulfide bonds. This package reprises my molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023), a structure-based program developed as part of my graduate thesis. The package relies on the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class to create and manipulate local coordinate systems. It does this by implementing the functions ``Move``, ``Roll``, ``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space.  The initial implementation focuses on the [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class. The class implements methods to analyze the protein structure stabilizing element known as a *Disulfide Bond*. This class and its underlying methods are being used to perform a structural analysis of over 36,900 disulfide-bond containing proteins in the RCSB protein data bank (https://www.rcsb.org).

# General Capabilities

- Interactively display disulfides contained in the RCSB in a variety of display styles
- Calculate geometric and energetic properties about these disulfides
- Create binary and octant structural classes by characterizing the disulfide torsional angles into *n* classes
- Build idealized disulfide bonds from dihedral angle input
- Find disulfide neighbors based on dihedral angle input
- Overlap disulfides onto a common frame of reference for display
- Build protein backbones from backbone phi, psi dihedral angle templates
- More in development

*See [API Reference](https://suchanek.github.io/proteusPy/proteusPy.html) for the API documentation with examples*.

# Requirements

1. PC running MacOS, Linux, Windows with git, git-lfs, make and C compiler installed.
2. 8 GB RAM
3. 1 GB disk space

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
  $ cd proteusPy
  $ make pkg
  $ conda activate proteusPy
  $ make install
  ```

## Windows

- Install Miniforge: <https://github.com/conda-forge/miniforge> (existing Anaconda installations are fine but please install mamba)
- Install git for Windows and configure for Bash:
  - https://git-scm.com/download/win
- Install git-lfs:
  - https://git-lfs.github.com/
- Install GNU make:
  - https://gnuwin32.sourceforge.net/packages/make.htm
- Open a Miniforge prompt and cd into your repo dir:
  ```console
  (base) C:\Users\egs\repos> git clone https://github.com/suchanek/proteusPy.git
  (base) C:\Users\egs\repos> cd proteusPy
  (base) C:\Users\egs\repos\proteuspy> make pkg
  (base) C:\Users\egs\repos>\proteuspy> conda activate proteusPy
  (proteusPy) C:\Users\egs\repos> make install
  ```

# Testing

I currently have ``pytest`` and docstring testing for the modules in place. To run them ``cd`` into the repository and run:
```console
$ make tests
```
The modules will run their docstring tests and disulfide visualization windows will open. Simply close them. If all goes normally there will be no errors. If you're not running the development version of proteusPy you may need to install ``pytest``. Simply perform: ``pip install pytest``. Docstring testing is sensitive to formatting; occasionally the ``black`` formatter changes the docstrings. As a result there may be some docstring tests that fail.

# Usage

Once the package is installed it's possible to load, visualize and analyze the Disulfide bonds in the RCSB Disulfide database. The general approach is:

- Load the database
- Access disulfide(s)
- Analyze
- Visualize

A simple example is shown below:

```python
import proteusPy
from proteusPy import Load_PDB_SS, Disulfide

PDB_SS = Load_PDB_SS(verbose=True)

best_ss = PDB_SS["2q7q_75D_140D"]
best_ss.display(style="sb", light=True)
```


The [notebooks](https://github.com/suchanek/proteusPy/blob/master/notebooks/) directory contains my Jupyter notebooks and is a good place to start:

- [Analysis_2q7q.ipynb](https://github.com/suchanek/proteusPy/blob/master/notebooks/Analysis_2q7q.ipynb) provides an example of visualizing the lowest energy Disulfide contained in the database and searching for nearest neighbors on the basis of conformational similarity.
- [Anearest_relatives.ipynb](https://github.com/suchanek/proteusPy/blob/master/notebooks/Anearest_relatives.ipynb) gives an example of searching for disulfides based on sequence similarity.

The [programs](https://github.com/suchanek/proteusPy/tree/master/programs) subdirectory contains the primary programs for downloading the RCSB disulfide-containing structure files, extracting the disulfides and creating the disulfide database:

- [DisulfideDownloader.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideDownloader.py): Downloads the raw RCSB structure files. The download consists of over 35,000 .ent files and took about twelve hours on a 200Mb internet connection. It is necessary to have these files locally to build the database. The download is about 35GB in size.
- [DisulfideExtractor_mp.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideExtractor_mp.py): Extracts the disulfides and creates the database loaders. This program is fully multi-processing, and one can specify the number of cores to use for the extract. The downloaded PDB files must be in $PDB/good. On my 14 core MacbookPro M3 Max the extraction of over 36,000 files and creation of the Disulfide loaders takes a bit over two minutes. This is in contrast to the initial single threaded version present in the initial release, which takes almost an hour to run!
- [DisulfideClass_Analysis.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideClass_Analysis.py): Performs binary or octant analysis on the disulfide database.

The first time one loads the database via [Load_PDB_SS()](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html#Load_PDB_SS) the system will attempt to download the full and subset database from Google Drive.

## Quickstart

After installation is complete launch jupyter:

```console
$ jupyter notebook 
```

and open ``notebooks/Analysis_2q7q.ipynb``. This notebook looks at the disulfide bond with the lowest energy in the entire database. There are several other notebooks in this directory that illustrate using the program. Some of these reflect active development work so may not be 'fully baked'.

## Visualizing Disulfides with pyVista

PyVista is an excellent 3D visualization framework and I've used it for the Disulfide visualization engine. It uses the VTK library on the back end and provides high-level access to 3d rendering. The menu strip provided in the Disulfide visualization windows allows the user to turn borders, rulers, bounding boxes on and off and reset the orientations. Please try them out! There is also a button for *local* vs *server* rendering. *Local* rendering is usually much smoother. To manipulate:

- Click and drag your mouse to rotate
- Use the mouse wheel to zoom (3 finger zoom on trackpad)

## Performance

- Manipulating and searching through long lists of disulfides can take time. I've added progress bars for many of these operations.
- Rendering many disulfides in **pyvista** can also take time to load and may be slow to display in real time, depending on your hardware. I added optimization to reduce cylinder complexity as a function of total cylinders rendered, but it can still be less than perfect. The faster your GPU the better!

## Contributing/Reporting

I welcome anyone interested in collaborating on proteusPy! Feel free to contact me at mailto:suchanek@mac.com, fork the repository: https://github.com/suchanek/proteusPy/ and get coding. Issues can be reported to https://github.com/suchanek/proteusPy/issues.

## Citing proteusPy

The proteusPy package was developed by Eric G. Suchanek, PhD. If you find it useful in your research and wish to cite it please use the following BibTeX entry:

```
@article{Suchanek2024,
  doi = {10.21105/joss.06169},
  url = {https://doi.org/10.21105/joss.06169},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {100},
  pages = {6169},
  author = {Eric G. Suchanek},
  title = {proteusPy: A Python Package for Protein Structure and Disulfide Bond Modeling and Analysis},
  journal = {Journal of Open Source Software}
}
@software{proteusPy2024,
  author = {Eric G. Suchanek, PhD},
  title = {proteusPy: A Package for Modeling and Analyzing Proteins of Known Structure},
  year = {2024},
  publisher = {GitHub},
  version = {0.96},
  journal = {GitHub repository},
  url = {https://github.com/suchanek/proteusPy}
}
```

## Publications

- [proteusPy: A Python Package for Protein Structure and Disulfide Bond Modeling and Analysis](https://joss.theoj.org/papers/10.21105/joss.06169)
- [Computer-aided Strategies for Protein Design](https://doi.org/10.1021/bi00368a023)
- [An engineered intersubunit disulfide enhances the stability and DNA binding of the N-terminal domain of .lambda. repressor](https://doi.org/10.1021/bi00368a024)
- [Introduction of intersubunit disulfide bonds in the membrane-distal region of the influenza hemagglutinin abolishes membrane fusion activity](https://doi.org/10.1016/0092-8674(92)90140-8)
- [Analysis of disulfide bonds in protein structures](http://dx.doi.org/10.1111/j.1538-7836.2010.03894.x)
