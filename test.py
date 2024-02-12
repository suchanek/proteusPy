'''
    This function creates .pkl files needed for the 
    proteusPy.DisulfideLoader.DisulfideLoader class. 
    The ```Disulfide``` objects are contained in a ```DisulfideList``` object and 
    ```Dict``` within these files. In addition, .csv files containing all of 
    the torsions for the disulfides and problem IDs are written. The optional
    ```dist_cutoff``` allows for removal of Disufides whose Cα-Cα distance is >
    than the cutoff value. If it's -1.0 then the function keeps all Disulfides.

    :param numb:           number of entries to process, defaults to all
    :param verbose:        more messages
    :param quiet:          turns off DisulfideConstruction warnings
    :param pdbdir:         path to PDB files
    :param datadir:        path to resulting .pkl files
    :param picklefile:     name of the disulfide .pkl file
    :param torsionfile:    name of the disulfide torsion file .csv created
    :param problemfile:    name of the .csv file containing problem ids
    :param dictfile:       name of the .pkl file
    :param dist_cutoff:    Ca distance cutoff to reject a Disulfide.
    
    The following examples illustrate some basic functions of the disulfide classes:

    >>> from proteusPy.Disulfide import Disulfide 
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList
        
    Instantiate some variables. Note: the list is initialized with an iterable and a name (optional)

    >>> SS = Disulfide('tmp')
    >>> SSlist = DisulfideList([],'ss')
    
    Load the Disulfide subset database. This contains around 8300 disulfides and loads
    fairly quickly.

    >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)

    The dataset can be indexed numerically, up to index: PDB_SS.Length(). Get the first SS:
    >>> SS = PDB_SS[0]
    >>> SS
    <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>

    The dataset can also be indexed by PDB ID. Get the DisulfideList for ID 4yys:

    >>> SS4yys = PDB_SS['4yys']
    >>> SS4yys
    [<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]

    Make some empty disulfides:

    >>> ss1 = Disulfide('ss1')
    >>> ss2 = Disulfide('ss2')

    Make a DisulfideList containing ss1, named 'tmp':

    >>> sslist = DisulfideList([ss1], 'tmp')

    Append ss2:
    >>> sslist.append(ss2)

    Extract the first disulfide and print it:

    >>> ss1 = PDB_SS[0]
    >>> ss1.pprint_all()
    <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å
     Proximal Chain fullID: <('4yys', 0, 'A', (' ', 22, ' '))> Distal Chain fullID: <('4yys', 0, 'A', (' ', 65, ' '))> 
    Proximal Coordinates:
       N: <Vector -2.36, -20.48, 5.21>
       Cα: <Vector -2.10, -19.89, 3.90>
       C: <Vector -1.12, -18.78, 4.12>
       O: <Vector -1.30, -17.96, 5.03>
       Cβ: <Vector -3.38, -19.31, 3.32>
       Sγ: <Vector -3.24, -18.40, 1.76>
       Cprev <Vector -2.67, -21.75, 5.36>
       Nnext: <Vector -0.02, -18.76, 3.36>
     Distal Coordinates:
       N: <Vector -0.60, -18.71, -1.62>
       Cα: <Vector -0.48, -19.10, -0.22>
       C: <Vector 0.92, -19.52, 0.18>
       O: <Vector 1.10, -20.09, 1.25>
       Cβ: <Vector -1.48, -20.23, 0.08>
       Sγ: <Vector -3.22, -19.69, 0.18>
       Cprev <Vector -0.73, -17.44, -2.01>
       Nnext: <Vector 1.92, -19.18, -0.63>
    <BLANKLINE>
     Proximal Internal Coords:
       N: <Vector -0.41, 1.40, -0.00>
       Cα: <Vector 0.00, 0.00, 0.00>
       C: <Vector 1.50, 0.00, 0.00>
       O: <Vector 2.12, 0.71, -0.80>
       Cβ: <Vector -0.50, -0.70, -1.25>
       Sγ: <Vector 0.04, -2.41, -1.50>
       Cprev <Vector -2.67, -21.75, 5.36>
       Nnext: <Vector -0.02, -18.76, 3.36>
    Distal Internal Coords:
       N: <Vector 1.04, -5.63, 1.17>
       Cα: <Vector 1.04, -4.18, 1.31>
       C: <Vector 1.72, -3.68, 2.57>
       O: <Vector 1.57, -2.51, 2.92>
       Cβ: <Vector -0.41, -3.66, 1.24>
       Sγ: <Vector -1.14, -3.69, -0.43>
       Cprev <Vector -0.73, -17.44, -2.01>
       Nnext: <Vector 1.92, -19.18, -0.63>
     Χ1-Χ5: 174.63°, 82.52°, -83.32°, -62.52° -73.83°, 138.89°, 1.70 kcal/mol
     Cα Distance: 4.50 Å
     Torsion length: 231.53 deg>
    

    Get a list of disulfides via slicing and display them oriented against a common
    reference frame (the proximal N, Cα, C').

    >>> subset = DisulfideList(PDB_SS[0:10],'subset')
    >>> subset.display_overlay()

    Take a screenshot. You can position the orientation, then close the window:
    >>> subset.screenshot(style='sb', fname='subset.png')
    ---> screenshot(): Saving file: subset.png
    ---> screenshot(): Saved file: subset.png

    Browse the documentation for more functionality. The display functions are particularly useful.
    '''


if __name__ == "__main__":
    import doctest
    doctest.testmod()
