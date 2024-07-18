---
title: 'proteusPy: A Python Package for Protein Structure and Disulfide Bond Modeling and Analysis'
tags:
  - Python
  - Disulfide Bonds
  - Protein Structure
  - RCSB Protein Databank
author: "Eric G. Suchanek, PhD."
authors:
  - name: Eric G Suchanek
    orcid: 0009-0009-0891-1507
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Flux-Frontiers, Cincinnati OH, United States of America
   index: 1
date: 8 July, 2024
header-includes:
  - \let\oldAA\AA
  - \renewcommand{\AA}{\text{\normalfont\oldAA}}
bibliography: joss.bib
---

<!-- markdownlint-disable MD014 -->
<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD037 -->
# Summary

**proteusPy** is a Python package specializing in the modeling and analysis of proteins of known structure with an initial focus on Disulfide bonds. This package significantly extends the capabilities of the molecular modeling program **proteus**, [@Pabo_1986], and utilizes a new implementation of the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class for disulfide and protein modeling.  This initial implementation focuses on the [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class, which implements methods to analyze the protein structure stabilizing element known as a **Disulfide Bond**, [DOI](https://zenodo.org/doi/10.5281/zenodo.11148440). 

The work has resulted in a freely-accessible database of over 120,494 disulfide bonds contained within 35,818 proteins in the [RCSB Protein Databank.](https:/www.rcsb.org) The routines within the library are capable of extracting, comparing, and visualizing the disulfides contained within the database, facilitating analysis and understanding. In addition, the package can readily model disulfide bonds of arbitrary conformation, facilitating predictive analysis.

# General Capabilities
- Interactively display disulfides contained in the RCSB in a variety of display styles
- Calculate geometric and energetic properties about these disulfides
- Create binary and sextant structural classes by characterizing the disulfide torsional angles into *n* classes
- Build idealized disulfide bonds from disulfide dihedral angle input
- Find disulfide neighbors based on dihedral angles
- Overlap disulfides onto a common frame of reference for display
- Build protein backbones from backbone phi, psi dihedral angle templates
- More in development

*See https://suchanek.github.io/proteusPy/proteusPy.html for the API documentation with examples*

# Statement of Need

Disulfide bonds, formed when two Cysteine residues are oxidized resulting in a sulfur-sulfur covalent bond, play pivotal roles in structural stabilization within and between protein subunits. Moreover, they participate in enzymatic catalysis, regulate protein activities, and offer protection against oxidative stress. Establishing an accessible structural database of these disulfides would serve as an invaluable resource for exploring these critical structural elements. While the capability to visualize protein structures is well established with excellent protein visualization tools like Pymol, Chimera and the RCSB itself, the tools for disulfide bond analysis are more limited. [@Wong_2010] describe a web-based disulfide visualization tool; this is currently unavailable.

Accordingly, I have developed the **proteusPy** package to delve into the RCSB Protein Data Bank, furnishing tools for visualizing and analyzing the disulfide bonds contained therein. This endeavor necessitated the creation of a python-based package containing data structures and algorithms capable loading, manipulating and analyzing these entities. Consequently, an object-oriented database has been crafted, facilitating introspection, analysis, and display. The package's API is accessible online at: [proteusPy API](https://suchanek.github.io/proteusPy/proteusPy.html), offering comprehensive details and numerous illustrative examples.


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

* [DisulfideDownloader.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideDownloader.py): Downloads the raw RCSB structure files.
* [DisulfideExtractor.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideExtractor.py): Extracts the disulfides and creating the database loaders
* [DisulfideClass_Analysis.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideClass_Analysis.py): Performs binary or sextant analysis on the disulfide database.

The first time one loads the database via [Load_PDB_SS()](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html#Load_PDB_SS) the system will attempt to download the full and subset database from Google Drive. If this fails it's possible to rebuild the database from the repo's **data** subdirectory (not the package's) by: ``pip install -e .`` at the repository top-level. If you've downloaded from github this will work correctly. If you've installed from pyPi via **pip** it will fail.


## Quickstart

After installation is complete, launch jupyter lab:

```console
$ jupyter notebook
```
and open [Analysis_2q7q](https://github.com/suchanek/proteusPy/blob/master/notebooks/Analysis_2q7q.ipynb). This notebook analyzes the disulfide bond with the lowest energy in the entire database and performs some searches in dihedral angle space to find similar conformations. There are several other notebooks in this directory that illustrate using the program. Some of these reflect active development work so may not be 'fully baked'.

# Class Details

The primary classes developed for **proteusPy** are described briefly below. Please see the [API](https://suchanek.github.io/proteusPy/proteusPy.html) for details.

# [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html)

This class provides a Python object and methods representing a physical disulfide bond either extracted from the RCSB protein databank or a virtual one built using the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class. The disulfide bond is an important intramolecular stabilizing structural element and is characterized by:

- Atomic coordinates for the atoms $N, C_{\alpha}$, $C_{\beta}$, $C'$, $S_\gamma$ for both amino acid residues. These are stored as both raw atomic coordinates as read from the RCSB file and internal local coordinates.
- The dihedral angles $\chi_{1} - \chi_{5}$ for the disulfide bond
- A name, by default: {pdb_id}{prox_resnumb}{prox_chain}_{distal_resnum}{distal_chain}
- Proximal residue number
- Distal residue number
- Approximate bond torsional energy (kcal/mol):
  $$
    E_{kcal/mol} \approx 2.0 * cos(3.0 * \chi_{1}) + cos(3.0 * \chi_{5}) + cos(3.0 * \chi_{2}) +
  $$
  $$
    cos(3.0 * \chi_{4}) + 3.5 * cos(2.0 * \chi_{3}) + 0.6 * cos(3.0 * \chi_{3}) + 10.1
  $$
- Euclidean length of the dihedral angles (degrees) defined as:
$$\sqrt(\chi_{1}^{2} + \chi_{2}^{2} + \chi_{3}^{2} + \chi_{4}^{2} + \chi_{5}^{2})$$
- $C_{\alpha} - C_{\alpha}$ distance ($\AA$)
- $C_{\beta} - C_{\beta}$ distance ($\AA$)
- The previous C' and next N coordinates for both the proximal and distal residues. These are needed to calculate the backbone dihedral angles $\phi$, $\psi$.
- Backbone dihedral angles $\phi$ and $\psi$, when possible. Not all structures are complete and in those cases the atoms needed may be undefined. In this case the $\phi$ and $\psi$ angles are set to -180°.

The class also provides 3D rendering capabilities using the excellent [PyVista](https://pyvista.org) library, and can display disulfides interactively in a variety of display styles:

- 'sb' - Split Bonds style - bonds colored by their atom type
- 'bs' - Ball and Stick style - split bond coloring with small atoms
- 'pd' - Proximal/Distal style - bonds colored *Red* for proximal residue and *Green* for the distal residue.
- 'cpk' - CPK style rendering, colored by atom type:

  - Carbon   - Grey
  - Nitrogen - Blue
  - Sulfur   - Yellow
  - Oxygen   - Red
  - Hydrogen - White

Individual renderings can be saved to a file and animations can be created. The *cpk* and *bs* styles are illustrated below:

![CPK & BS Disulfide Rendering](bs_cpk.png)

# [DisulfideLoader](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html)

This class encapsulates the disulfide database itself and is its primary means of accession.  Instantiation takes 2 parameters: **subset** and **verbose**. Given the size of the database, one can use the **subset** parameter to load the first 1000 disulfides into memory. This facilitates quicker development and testing new functions. I recommend using a machine with 16GB or more to work with the full dataset.

The entirety of the RCSB disulfide database is stored within the class via a [DisulfideList]("https://suchanek.github.io/proteusPy/proteusPy/DisulfideList.html"), a **Pandas** .csv file, and a **dict** of indices mapping the RCSB IDs into their respective list of disulfide bond objects. The datastructures allow simple, direct and flexible access to the disulfide structures contained within. This makes it possible to access the disulfides by array index, RCSB structure ID or disulfide name.

Example:

```
  import proteusPy
  from proteusPy import Disulfide, DisulfideLoader, DisulfideList

  SS1 = DisulfideList([],'tmp1')
  SS2 = DisulfideList([],'tmp2')

  PDB_SS = DisulfideLoader(verbose=False, subset=True)

  # Accessing by index value:
  SS1 = PDB_SS[0]
  SS1
  <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>

  # Accessing by PDB_ID returns a list of Disulfides:
  SS2 = PDB_SS['4yys']
  SS2
  [<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, 
  <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, 
  <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]

  # Accessing individual disulfides by their name:
  SS3 = PDB_SS['4yys_56A_98A']
  SS3
  <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>

  # Finally, we can access disulfides by regular slicing:
  SSlist = SS2[:2]
  [<Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, 
  <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]
```

The class can also render Disulfides overlaid on a common coordinate system to a pyVista window using the [DisulfideLoader.display_overlay()](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html#DisulfideLoader.display_overlay) method.

**NB:** For typical usage one accesses the database via the **Load_PDB_SS()** function. This function loads the compressed database from its single source. Initializing a **DisulfideLoader** object will load the individual torsions and disulfide **.pkl** files, builds the classlist structures, and writes the completely built object to a single **.pkl** file. This requires the raw **.pkl** files created by the download process. These files are contained in the *repository* **data** directory, not in the **pyPi** distribution.

# [turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html)

The **turtle3D** class represents an object that maintains a *local coordinate system* in three dimensional space. This coordinate system consists of:

- A Position in 3D space 
- A Heading Vector
- A Left Vector
- An Up Vector

The *Heading*, *Left* and *Up* vectors are unit vectors that define the 
object's orientation in a *local* coordinate frame. The turtle developed in **proteusPy** is based on the excellent book by Abelson: [@Abelson_DiSessa_1986]. The [to_local]("https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html#Turtle3D.to_local") and [to_global]("https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html#Turtle3D.to_global") methods convert between these two coordinate systems. These methods make it possible to readily compare different disulfides by:

1. Orienting the turtle at the disulfide's proximal residue in a standard orientation.
2. Converting the global coordinates of the disulfide as read from the RCSB into local coordinates.
3. Saving all of the local coordinates with the raw coordinates
4. Performing distance and angle calculations

By implementing the functions **Move**, **Roll**, **Yaw**, **Pitch** and **Turn** the turtle is capable of movement in a three-dimensional space. See [@Pabo_1986] for more details.

The turtle has several molecule-specific functions including [orient_at_residue]("https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html#Turtle3D.orient_at_residue") and [orient_from_backbone]("https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html#Turtle3D.orient_from_backbone"). These routines make it possible to build protein backbones of arbitrary conformation and to readily add sidechains to modeled structures. These functions are currently used to build model disulfides from dihedral angle input.


# Examples

I illustrate a few use cases for the package below. Use the **jupyter notebook** command from your shell to launch jupyter. The examples illustrate the ease with which one can analyze and visualize some disulfides. 

## Find the lowest and highest energy disulfides present in the database

```python
from proteusPy import Load_PDB_SS, DisulfideList, Disulfide

# load the database
PDB_SS = Load_PDB_SS(verbose=True, subset=False)

# retrieve the minimum and maximum energy structures
ssMin, ssMax = PDB_SS.SSList.minmax_energy

# make a list to hold them
minmaxlist = DisulfideList([ssMin, ssMax], "minmax")

# display them as ball and stick style
minmaxlist.display(style="bs", light=True)
```

<center>

![minmax](minmax.png)

</center>

## Find disulfides within 10 $\AA$ RMS in torsional space of the lowest energy structure

In this example we load the disulfide database, find the disulfides with
the lowest and highest energies, and then find the nearest conformational neighbors. Finally, we display the neighbors overlaid against a common reference frame. Note that the window title gives statistics about the list of disulfides being displayed, including list name, resolution, number, average energy, and average atom positional error.

```python
import proteusPy
from proteusPy Load_PDB_SS, DisulfideList, Disulfide

PDB_SS = None
PDB_SS = Load_PDB_SS(verbose=False, subset=False)
ss_list = DisulfideList([], "tmp")

# Return the minimum and maximum energy structures. We ignore the maximum in this case.
ssmin_enrg, _ = PDB_SS.SSList.minmax_energy

# Make an empty list and find the nearest neighbors within 10 degrees avg RMS in
# sidechain dihedral angle space.

low_energy_neighbors = DisulfideList([], "Neighbors")
low_energy_neighbors = ssmin_enrg.Torsion_neighbors(sslist, 10)

# Display the number found, and then display them overlaid onto their common reference frame.

tot = low_energy_neighbors.length
low_energy_neighbors.display_overlay()
```

18

<center>

![Low energy neighbors](min_overlay.png)
</center>

# Analyzing Disulfide Structural Class Distributions

The package includes the [DisulfideClassConstructer](https://suchanek.github.io/proteusPy/proteusPy/DisulfideClass_Constructor.html) class, which is used to create and manage Disulfide binary and sextant classes. A note about these structural classes is in order. [@Schmidt_2006] described a method of characterizing disulfide structures by describing each individual dihedral angle as either + or - based on its sign. This yields $2^{5}$ or 32 possible classes. The author was then able to classify protein functional families into one of 20 remaining structural classes. Since the binary approach is very coarse and computers  are much more capable than in 2006 I extended this formalism to a *Sextant* approach. In other words, I created *six* possible classes for each dihedral angle by dividing it into 60 degree segments. This yields a possible $6^5$ or 7,776 possible classes. The notebook [DisulfideClassesPlayground.ipynb](https://github.com/suchanek/proteusPy/blob/master/notebooks/DisulfideClassesPlayground.ipynb) contains some initial results. This work is ongoing.

# Summary

**proteusPy** is a python-based package capable of visualization and analysis of over 120,000 Disulfide bonds contained in the RCSB structural database. This work provides a strong foundation to not only analyze these important structural elements but also provides flexible tools for modeling proteins from dihedral angle input. 


## Contributing/Reporting

I welcome anyone interested in collaborating on proteusPy! Feel free to contact me at suchanek@mac.com, fork the [repo](https://github.com/suchanek/proteusPy/) and get coding. Issues can be reported to https://github.com/suchanek/proteusPy/issues. 

# Bibliography
