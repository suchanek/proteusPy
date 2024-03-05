# Analysis of Disulfide Bonds in Proteins Within the RCSB Protein Data Bank
*Eric G. Suchanek, PhD. (suchanek@mac.com)* <br> <br>

## Summary
I describe the results of a structural analysis of Disulfide bonds contained in 35,819 proteins within the RCSB Protein databank, https://www.rcsb.org. These protein structures contained 120,697 Disulfide Bonds.  The analysis utilizes Python functions from my ``ProteusPy`` package https://github.com/suchanek/proteusPy/, which is built upon the excellent ``BioPython`` library (https://www.biopython.org). 

## Background
This work represents a reprise of my original Disulfide modeling analysis conducted in 1986 ([publications](#publications) item 1) as part of my doctoral dissertation. Given the original Disulfide database contained only 2xx Disulfide Bonds I felt it would be interesting to revisit the RCSB and mine the thousands of new structures. The analysis would not have been possible without the creation and now ongoing development of ```proteusPy```, which represents a modern object-oriented rewrite of my original ```proteus``` code base. The ```C``` implementation still exists and is available at https://github.com/suchanek/proteus/.


## Intended Audience
This notebook is intended primarily for people interested in structural biophysics, with an emphasis on the analysis of protein structures. As a result, it is fairly advanced. I have made an effort to describe the analysis plainly, but I assume the reader has a basic understanding of the elements of protein structure.

## Requirements
 - My Biopython fork or my delta applied, available at: https://github.com/suchanek/biopython/
 - proteusPy: https://github.com/suchanek/proteusPy/

## Introduction
Disulfide bonds are important covalent stabilizing elements in proteins, and function as intra-molecular cross-bridges. They are formed when two Sulphur-containing Cysteine (Cys) amino acid residues are close enough and in the correct geometry to form a S-S covalent bond with their terminal sidechain $S_\gamma$ atoms. The resulting residue is known as *Cystine*, or more commonly a *Disulfide* bond. This cross-bridge stabilizes the connected protein backbone via the alpha carbon, ($C_\alpha$) backbone atoms with this strong S-S covalent bond. This bond has high energy barriers to rotation, and as such has manifests partial double-bond character. Disulfide bonds most commonly occur between alpha helices and greatly enhance a protein's stability to denaturation. 

## Download Disulfides

## Extract the Disulfides from the PDB files
The function ``Extract_Disulfides()`` processes all the .ent files in ``PDB_DIR`` and creates two .pkl files representing the Disulfide bonds contained in the scanned directory. In addition, a .csv file containing problem IDs is written if any are found. The .pkl files are consumed by the ``DisulfideLoader`` class and are considered private. You'll see numerous warnings during the scan. Files that are unparsable are removed and their IDs are logged to the problem_id.csv file. The default file locations are stored in the file globals.py and are the used by ``DisulfideExtractor()`` in the absence of arguments passed. The Disulfide parser is very stringent and will reject disulfide bonds with missing atoms or disordered atoms.


Outputs are saved in ``MODEL_DIR``:
1) ``SS_PICKLE_FILE``: The ``DisulfideList`` of ``Disulfide`` objects initialized from the PDB file scan, needed by the ``DisulfideLoader()`` class.
2) ``SS_DICT_PICKLE_FILE``: the ``Dict Disulfide`` objects also needed by the ``DisulfideLoader()`` class
3) ``PROBLEM_ID_FILE``: a .csv containing the problem ids.

In general, the process only needs to be run once for a full scan. Setting the ``numb`` argument to -1 scans the entire directory. Entering a positive number allows parsing a subset of the dataset, which is useful when debugging. Setting ``verbose`` enables verbose messages. Setting ``quiet`` to ``True`` disables all warnings.

NB: A extraction of the initial disulfide bond-containing files (> 36000 files) takes about 1.25 hours on a 2020 MacbookPro with M1 Pro chip, 16GB RAM, 1TB SSD. The resulting .pkl files consume approximately 1GB of disk space, and equivalent RAM used when loaded.

## Load the Disulfide Data
Now that the Disulfides have been extracted and the Disulfide .pkl files have been created we can load them into memory using the DisulfideLoader() class. This class stores the Disulfides internally as a DisulfideList and a dict. Array indexing operations including slicing have been overloaded, enabling straightforward access to the Disulfide bonds, both in aggregate and by residue. After loading the .pkl files the Class creates a Pandas ``DataFrame`` object consisting of the Disulfide ID, all sidechain dihedral angles, the local coordinates for the Disulfide and the computed Disulfide bond torsional energy.

NB: Loading the data takes 3.5 minutes on my MacbookPro. Be patient if it seems to take a long time to load.

The ```Disulfide``` and ```DisulfideList``` classes include rendering capabilities using the excellent PyVista interface to the VTK package. (http://pyvista.org). The following cell displays the first Disulfide bond in the database in ball-and stick style. Atoms are colored by atom type:
- Grey = Carbon
- Blue = Nitrogen
- Red = Oxygen
- Yellow = Sulfur
- White = Previous residue carbonyl carbon and next residue amino Nitrogen. (more on this below).

We can load the database and display the first disulfide as shown below. The molecular display is *interactive*; select-drag to rotate, mousewheel to zoom. The X-Y-Z widget in the window upper right allows orientation against the X, Y and Z axes. The window title provides information about the disulfide rendered:
* Disulfide source ID
* Disulfide name, which embodies the source id, the proximal residue number and the distal residue number.
* Disulfide approximate torsional energy (kcal/mol)
* Disulfide $C_\alpha-C_\alpha$ distance $\AA$
* Disulfide torsion length (5-dimensional vector length) (degrees)

## Examine the Disulfide $C_\alpha-C_\alpha$ Distances

The disulfide bond maximum $C_\alpha-C_\alpha$ distance is constrained by the bond lengths and bond angles of the cystine (Cys-Cys) sidechain atoms. If the bond angles were linear the maximum possible $C_\alpha-C_\alpha$ distance would be 8.82 $\AA$. We can examine the database easily with Pandas to see this distance distribution.

Initial analysis shown above reveals 145 physically impossible disulfides. These will be removed from consideration by rejecting disulfides whose $C_\alpha-C_\alpha$ distances are $> 9 \AA$. This should improve our overall protein structure data quality.

When we filter by distance, we see the average $C_\alpha-C_\alpha$ distance for the entire dataset is 5.52 $\AA$ , with a minimum distance of 2.83 $\AA$ and a maximum of 8.50 $\AA$. As mentioned above, 145 disulfides had distances >= 9 $\AA$. These will be removed from consideration since they are not physically possible.

We can extract and visualize the four longest structures and display them, as shown below. The individual windows display the disulfide bond in 'split-bond' style, where half of the bond is colored by the respective atom color. The window title indicates the approximate torsional energy (E kcal/mol), $C_\alpha-C_\alpha$ distance ($\AA$), and the torsion length (degrees). As is apparent, the $C_\beta-S_\gamma$ bond angles are almost linear. This suggests that the disulfide might not actually exist covalently, and could reflect errors in the original model.

## Examine the Disulfide Torsions
The disulfide bond's overall conformation is defined by the sidechain dihedral angles $\chi_{1}$-$\chi_{5}$. Since the S-S bond has electron delocalization, it exhibits some double-bond character with strong minima at $+90°$ and $-90°$. The *Left-handed* Disulfides have $\chi_{3}$ < 0.0° and the *Right-handed* have a $\chi_{3}$ > 0.0°.

These torsion values along with the approximate torsional energy are stored in the DisulfideLoader() class and individually within each Disulfide object. We access them via the ``DisulfideList.getTorsions()`` function.

We can get a quick look at their overall statistics using the ``Pandas.describe()`` function.

### Examining torsions by Disulfide Handedness
We split the dataset into these two families easily with Pandas.


### Disulfide Family Analysis


[label](average_ss_byclass.md)
## Conclusions
Conformational analysis of 294,222 disulfide bonds in 36,362 proteins contained in the RCSB confirms the predominant conformational classes first described in my initial analysis:
- Left-Handed Spiral
- Right-Handed Hook
- Left-Handed Spiral
  
# Appendix
## Data Cleaning
* Parsing PDB files
* Extracting disulfides
* Removing redundant disulfides
* 
# Publications
* https://doi.org/10.1021/bi00368a023
* https://doi.org/10.1021/bi00368a024
* https://doi.org/10.1016/0092-8674(92)90140-8
* http://dx.doi.org/10.2174/092986708783330566