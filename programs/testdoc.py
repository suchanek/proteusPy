class test:
    """
    Class provides a sortable list for Disulfide objects.
    Indexing and slicing are supported, and normal list operations like .insert, .append and .extend.
    The DisulfideList object must be initialized with an iterable (tuple, list) and a name.

    The class can also render Disulfides to a pyVista window using the DisulfideList.display()
    method. See below for examples.\n

    Examples:
        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideLoader import DisulfideLoader
        >>> from proteusPy.DisulfideList import DisulfideList
        >>> from proteusPy.proteusGlobals import MODEL_DIR

        # instantiate some variables
        # Note: the list is initialized with an iterable and a name (optional)

        >>> SS = Disulfide('tmp')
        >>> SSlist = DisulfideList([],'ss')

        >>> PDB_SS = DisulfideLoader()  # load the Disulfide database\n
        >>> SS = PDB_SS[0]              # returns a Disulfide object at index 0
        >>> SSlist = PDB_SS['4yys']     # returns a DisulfideList containing all
                                        #  disulfides for 4yys\n

        >>> SSlist = PDB_SS[:8]         # get SS bonds for the last 8 structures\n
        >>> SSlist.display('sb')        # render the disulfides in 'split bonds' style\n

        # make some empty disulfides
        >>> ss1 = Disulfide('ss1')
        >>> ss2 = Disulfide('ss2')

        # make a DisulfideList containing ss1, named 'tmp'
        >>> sslist = DisulfideList([ss1], 'tmp')
        >>> sslist.append(ss2)

        # extract the first disulfide
        >>> ss1 = PDB_SS[0]
        >>> print(f'{ss1.pprint_all()}')

        # grab a list of disulfides via slicing
        >>> subset = DisulfideList(PDB_SS[0:10],'subset')
        >>> subset.display(style='sb')      # display the disulfides in 'split bond' style
        >>> subset.display_overlay()        # display all disulfides overlaid in stick style
        >>> subset.screenshot(style='sb', fname='subset.png')  # save a screenshot.
    """


from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import DisulfideLoader
from proteusPy.DisulfideList import DisulfideList
from proteusPy.ProteusGlobals import MODEL_DIR

PDB_ROOT = "/Users/egs/PDB/"

# location of cleaned PDB files - these are not stored in the repo
PDB_GOOD = "/Users/egs/PDB/good/"

# from within the repo
PDB_REPO = "../pdb/"

# location of the compressed Disulfide .pkl files
MODELS = f"{PDB_ROOT}models/"

# instantiate some variables
# Note: the list is initialized with an iterable and a name (optional)

SS = Disulfide("tmp")
SSlist = DisulfideList([], "ss")

PDB_SS = DisulfideLoader(verbose=True, datadir=MODELS)  # load the Disulfide database\n
SS = PDB_SS[0]  # returns a Disulfide object at index 0
SSlist = PDB_SS["4yys"]  # returns a DisulfideList containing all
#  disulfides for 4yys\n

SSlist = PDB_SS[:8]  # get SS bonds for the last 8 structures\n
SSlist.display("sb")  # render the disulfides in 'split bonds' style\n

# make some empty disulfides
ss1 = Disulfide("ss1")
ss2 = Disulfide("ss2")

# make a DisulfideList containing ss1, named 'tmp'
sslist = DisulfideList([ss1], "tmp")
sslist.append(ss2)

# extract the first disulfide
ss1 = PDB_SS[0]
print(f"{ss1.pprint_all()}")

# grab a list of disulfides via slicing
subset = DisulfideList(PDB_SS[0:10], "subset")
# subset.display(style='sb')      # display the disulfides in 'split bond' style
# subset.display_overlay()        # display all disulfides overlaid in stick style
# subset.screenshot(style='sb', fname='subset.png')  # save a screenshot.
