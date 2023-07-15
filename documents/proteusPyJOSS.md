---
title: 'proteusPy: A Python Package for Disulfide Bond Analysis'
tags:
  - Python
  - disulfide bonds
  - protein structure
  - RCSB protein databank
authors:
  - name: Eric G Suchanek
    orcid: 0009-0009-0891-1507
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Monterey Institute for Research in Astronomy, Marina, USA
   index: 1
date: 4 July, 2023
bibliography: proteusPyJOSS.bib
---

# Summary
*proteusPy* is a Python package specializing in the modeling and analysis of proteins of known structure with an emphasis on Disulfide Bonds. This package reprises my molecular modeling program *proteus*, [@Pabo_1986], and utilizes a new implementation of the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class for disufulfide modeling. The turtle implements the functions ``Move``, ``Roll``, ``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space. The [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class implements methods to analyze the protein structure stabilizing element known as a *Disulfide Bond*. This class and its underlying methods are being used to perform a structural analysis of over 35,800 disulfide-bond containing proteins in the RCSB protein data bank.

# Virtual Environment Creation
1. *Install Anaconda (<http://anaconda.org>)*
2. *Build the environment.* 
   At this point it's probably best to clone the repo via github since it contains all
   of the notebooks test programs and raw Disulfide databases. The source code distribution can be used from pyPi as a normal
   package, within your own environment.
   - Using pyPi:
     - python3 -m pip install proteusPy
   - From the gitHub repository:
     - Install git-lfs
       - https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage
       - From a shell prompt: 
         ```
          $ git-lfs track "*.csv" "*.pkl" "*.mp4"
          $ git clone https://github.com/suchanek/proteusPy/proteusPy.git
          $ cd proteusPy
          $ conda env create --name proteusPy --file=proteusPy.yml
          $ conda activate proteusPy
          $ pip install .
          $ jupyter nbextension enable --py --sys-prefix widgetsnbextension

         ```

# General Usage
Once the package is installed one can use the existing notebooks for analysis of the RCSB Disulfide database. The ``notebooks`` directory contains all of my Jupyter notebooks and is a good place to start. The ``DisulfideAnalysis.ipynb`` notebook contains the first analysis paper. The ``programs`` subdirectory contains the primary programs for downloading the RCSB disulfide-containing structure files, (``DisulfideDownloader.py``), extracting the disulfides and creating the database loaders (``DisulfideExtractor.py``) and cluster analysis (``DisulfideClass_Analysis.py``).

The first time one loads the database via ``Load_PDB_SS()`` the system will attempt to download the full and subset database from my Google Drive. If this fails the system will attempt to rebuild the database from the repo's ``data`` subdirectory (not the package's). If you've downloaded from github this will work correctly. If you've installed from pyPi via ``pip`` it will fail.

# The Class Details
The primary driver for implementing ``proteusPy`` was to revisit the [RCSB Protein Databank](https://www.rcsb.org) and do a structural analysis of the disulfide bonds contained therein. This necessitated the creation an object-oriented database capable of introspection analysis, and display. I'll describe the primary classes below. The API is available online at: https://suchanek.github.io/proteusPy/proteusPy.html.

##[Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html)


##[DisulfideLoader](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html)

This class represents the disulfide database itself and is its primary means of accession.  Instantiation takes 2 parameters: ``subset`` and ``verbose``. Given the size of the database, one can use the ``subset`` parameter to load the first 1000 disulfides into memory. This facilitates quicker development and testing new functions. I recommend using at least a 16 GB machine to work with the full dataset.

The entirety of the RCSB disulfide database is stored within the class via a [proteusPy.DisulfideList]("https://suchanek.github.io/proteusPy/proteusPy/DisulfideList.html"), a ```Pandas``` .csv file, and a ```dict``` of indices mapping the PDB IDs into their respective list of disulfides. The datastructures allow simple, direct and flexible access to the disulfide structures contained within. This makes it possible to access the disulfides by array index, PDB structure ID or disulfide name.

Example:

    >>> import proteusPy
    >>> from proteusPy.Disulfide import Disulfide
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList
    >>> SS1 = DisulfideList([],'tmp1')
    >>> SS2 = DisulfideList([],'tmp2')
    
    >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)

    Accessing by index value:
    >>> SS1 = PDB_SS[0]
    >>> SS1
    <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>
    
    Accessing by PDB_ID returns a list of Disulfides:
    >>> SS2 = PDB_SS['4yys']
    >>> SS2
    [<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]
    
    Accessing individual disulfides by their name:
    >>> SS3 = PDB_SS['4yys_56A_98A']
    >>> SS3
    <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>
    
    Finally, we can access disulfides by regular slicing:
    >>> SSlist = PDB_SS[:4]

The class can also render Disulfides overlaid on a common coordinate system to a pyVista window using the [DisulfideLoader.display_overlay()](https://suchanek.github.io/proteusPy/proteusPy/DisulfideLoader.html#DisulfideLoader.display_overlay) method. 

**NB:** For typical usage one accesses the database via the `Load_PDB_SS()` function. This function loads the compressed database from its single source. Initializing a `DisulfideLoader()` object will load the individual torsions and disulfide .pkl files, builds the classlist structures, and writes the completely built object to a single ``.pkl`` file. This requires the raw .pkl files created by download process. These files are contained in the repository ``data`` directory.

*Developer's Notes:*
The .pkl files needed to instantiate this class and save it into its final .pkl file are
defined in the [proteusPy.data]("https://suchanek.github.io/proteusPy/proteusPy/data.html") class and should not be changed. Upon initialization the class
will load them and initialize itself. 



  * [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html)
  * 
  

to manage both the representation of the 

This resulted in the following overall workflow:
* Identify disulfide containing proteins in the [RCSB](https://www.rcsb.org). I generated a query using their web-based query tool. The resulting file consisted of 35819 proteins containing over 200,000 disulfide bonds.
* Download the structure files to disk. This resulted in the program ``DisulfideDownloader.py``. 
* Extract the disulfides from the downloaded structures:
  * Check for structural feasibility
  * 
# The Future
I am continuing to explore the initial disulfide structural classes gleaned from Hogg *et al.* further using the sextant class approach. This offers much higher class resolution and reveals subgroups within the broad class. I'd also like to explore the catalytic and allosteric classes in more detail to look for common structural elements.

# Citations
* [Pabo_1986]
* [proteusPy]


*NB:* This distribution is being developed slowly. proteusPy relies on my fork of the ``Bio`` Python package to download and build the database. As a result, one can't download and create the database locally unless the BioPython patch is applied. The changed python file is in the repo's data directory - ``parse_pdb_header.py``. Database analysis is unaffected without the patch. Also, if you're running on an M-series Mac then it's important to install Biopython first, since the generic release won't build on the M1. 7/4/23 -egs-

Eric G. Suchanek, PhD., <mailto:suchanek@mac.com>

