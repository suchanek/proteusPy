# DisulfideList Class definition
# I extend UserList to handle lists of Disulfide objects.
# Indexing and slicing are supported, sorting is based on energy
# Author Eric G. Suchanek, PhD
# A part of the proteusPy molecular modeling and analysis suite by
# Eric G. Suchanek, PhD
# Last modification: 12/13/2023 -egs-

import pandas as pd
import tqdm

import proteusPy
from proteusPy import *
from proteusPy.atoms import *

import pyvista as pv
from collections import UserList
from tqdm import tqdm

_PBAR_COLS = 100

def grid_dimensions(n):
    '''
    Calculate rows and columns for the given needed to display
    a given number of disulfides in a square aspect.

    :param n: int Number of Disulfides
    :return: int rows, columns
    '''
    
    root = math.sqrt(n)
    # If the square root is a whole number, return that as the number of rows and columns
    if root == int(root):
        return int(root), int(root)
    # If the square root is not a whole number, round up and return that as the number of columns
    # and calculate the number of rows as the number of images divided by the number of columns
    else:
        columns = math.ceil(root)
        return int(n / columns), int(columns)

Torsion_DF_Cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', \
           'chi5', 'energy', 'ca_distance', 'phi_prox', 'psi_prox', 'phi_dist',\
           'psi_dist', 'torsion_length']

Distance_DF_Cols = ['source', 'ss_id', 'proximal', 'distal', 'energy', 'ca_distance']

class DisulfideList(UserList):
    '''
    Class provides a sortable list for Disulfide objects.
    Indexing and slicing are supported, and normal list operations like 
    .insert, .append and .extend. The DisulfideList object must be initialized 
    with an iterable (tuple, list) and a name.
    
    The class can also render Disulfides to a pyVista window using the 
    DisulfideList.display() method. See below for examples.\n

    Examples:
    >>> from proteusPy.Disulfide import Disulfide 
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList
        
    # instantiate some variables
    # Note: the list is initialized with an iterable and a name (optional)

    >>> SS = Disulfide('tmp')
    >>> SSlist = DisulfideList([],'ss')

    >>> PDB_SS = DisulfideLoader(verbose=True, subset=True)  # load the Disulfide database
    Reading disulfides from: /Users/egs/repos/proteusPy/proteusPy/data/PDB_all_ss.pkl
    Disulfides Read: 8210
    Reading disulfide dict from: /Users/egs/repos/proteusPy/proteusPy/data/PDB_all_ss_dict.pkl
    Reading Torsion DF /Users/egs/repos/proteusPy/proteusPy/data/PDB_all_SS_torsions.csv.
    Read torsions DF.
    PDB IDs parsed: 1000
    Total Space Used: 1969317 bytes.

    >>> SS = PDB_SS[0]              # returns a Disulfide object at index 0
    >>> SS
    <Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A>
    >>> SS4yys = PDB_SS['4yys']     # returns a DisulfideList containing all
    >>> SS4yys
    [<Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A>, <Disulfide 4yys_56A_98A SourceID: 4yys Proximal: 56 A Distal: 98 A>, <Disulfide 4yys_156A_207A SourceID: 4yys Proximal: 156 A Distal: 207 A>, <Disulfide 4yys_22B_65B SourceID: 4yys Proximal: 22 B Distal: 65 B>, <Disulfide 4yys_56B_98B SourceID: 4yys Proximal: 56 B Distal: 98 B>, <Disulfide 4yys_156B_207B SourceID: 4yys Proximal: 156 B Distal: 207 B>]

    # make some empty disulfides
    >>> ss1 = Disulfide('ss1')
    >>> ss2 = Disulfide('ss2')

    # make a DisulfideList containing ss1, named 'tmp'
    >>> sslist = DisulfideList([ss1], 'tmp')
    >>> sslist.append(ss2)

    # extract the first disulfide
    >>> ss1 = PDB_SS[0]
    >>> ss1.pprint_all()
    <Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A
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
     Conformation: (Χ1-Χ5):  174.629°, 82.518°, -83.322°, -62.524° -73.827°  Energy: 1.696 kcal/mol
     Ca Distance: 4.502 Å>

    # get a list of disulfides via slicing
    >>> subset = DisulfideList(PDB_SS[0:10],'subset')
    >>> subset.display_overlay()        # display all disulfides overlaid in stick style
    '''
    
    def __init__(self, iterable, id):
        self.pdb_id = id
        super().__init__(self.validate_ss(item) for item in iterable)

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.data)))
            name = self.data[0].pdb_id
            sublist = [self.data[i] for i in indices]
            return DisulfideList(sublist, name)    
        return UserList.__getitem__(self, item)
    
    def __setitem__(self, index, item):
        self.data[index] = self.validate_ss(item)

    def insert(self, index, item):
        self.data.insert(index, self.validate_ss(item))

    def append(self, item):
        self.data.append(self.validate_ss(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            self.data.extend(other)
        else:
            self.data.extend(self._validate_ss(item) for item in other)
    
    def validate_ss(self, value):
        if isinstance(value, (proteusPy.Disulfide.Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")
    
    @property
    def id(self):
        return(self.pdb_id)
    
    @id.setter
    def id(self, value):
        self.pdb_id = value
    
    def get_by_name(self, name):
        '''
        Returns the Disulfide with the given name from the list.
        '''
        sslist = self.data
        res = None

        for ss in sslist:
            id = ss.name
            if id == name:
                res = ss.copy()
                break
        return res

    def minmax_energy(self):
        """
        Return the Disulfides with the minimum and maximum energies
        from the DisulfideList.
        
        :return: Minimum, Maximum
        :rtype: Disulfide
        """
        sslist = sorted(self.data)
        return sslist[0], sslist[-1]

    def min(self):
        sslist = sorted(self.data)
        return sslist[0]
    
    def max(self):
        sslist = sorted(self.data)
        return sslist[-1]
    
    def get_chains(self):
        '''
        Return the chain IDs for chains within the given Disulfide.
        '''
        res_dict = {'xxx'}
        sslist = self.data

        for ss in sslist:
            pchain = ss.proximal_chain
            dchain = ss.distal_chain
            res_dict.update(pchain)
            res_dict.update(dchain)
        
        res_dict.remove('xxx')

        return res_dict

    def has_chain(self, chain) -> bool:
        '''
        Returns True if given chain contained in Disulfide, False otherwise.
        '''
        
        chns = {'xxx'}
        chns = self.get_chains()
        if chain in chns:
            return True
        else:
            return False

    def by_chain(self, chain: str):
        '''
        Return a DisulfideList from the input chain identifier.

        Arguments:
            chain: str - chain identifier, 'A', 'B, etc
        Returns:
            DisulfideList containing disulfides within that chain.
        '''
        
        reslist = DisulfideList([], chain)
        sslist = self.data

        for ss in sslist:
            pchain = ss.proximal_chain
            dchain = ss.distal_chain
            if pchain == dchain:
                if pchain == chain:
                    reslist.append(ss)
            else:
                print(f'Cross chain SS: {ss.repr_compact}:')
        return reslist
    
    @property
    def torsion_df(self):
        return self.build_torsion_df()
    
    def build_torsion_df(self) -> pd.DataFrame:
        '''
        Create a dataframe containing the input DisulfideList torsional parameters,
        ca-ca distance, energy, and phi-psi angles. This can take a while for the
        entire database.

        :param SSList: DisulfideList - input list of Disulfides
        :return: pandas.Dataframe containing the torsions
        '''
        # create a dataframe with the following columns for the disulfide 
        # conformations extracted from the structure
        
        SS_df = pd.DataFrame(columns=Torsion_DF_Cols)
        sslist = self.data

        pbar = tqdm(sslist, ncols=_PBAR_COLS)
        for ss in pbar:
            new_row = [ss.pdb_id, ss.name, ss.proximal, ss.distal, ss.chi1, ss.chi2, 
                    ss.chi3, ss.chi4, ss.chi5, ss.energy, ss.ca_distance,
                    ss.psiprox, ss.psiprox, ss.phidist, ss.psidist, ss.torsion_distance]
            # add the row to the end of the dataframe
            SS_df.loc[len(SS_df.index)] = new_row
        
        return SS_df

    @property
    def distance_df(self):
        return self.build_distance_df()
    
    def build_distance_df(self) -> pd.DataFrame:
        """
        Create a dataframe containing the input DisulfideList ca-ca distance, energy. 
        This can take several minutes for the entire database.

        :return: DataFrame containing Ca distances
        :rtype: pd.DataFrame
        """
        
        # create a dataframe with the following columns for the disulfide 
        # conformations extracted from the structure
        
        SS_df = pd.DataFrame(columns=Distance_DF_Cols)
        sslist = self.data

        pbar = tqdm(sslist, ncols=_PBAR_COLS)
        for ss in pbar:
            new_row = [ss.pdb_id, ss.name, ss.proximal, ss.distal, ss.energy, ss.ca_distance]
            # add the row to the end of the dataframe
            SS_df.loc[len(SS_df.index)] = new_row
        
        return SS_df

    def minmax_distance(self):
        """
        """
        distance_df = self.build_distance_df()
        distance_df.sort_values(by=['ca_distance'], ascending=True, inplace=True)
        ssmin_id = distance_df.iloc[0]['ss_id']
        ssmax_id = distance_df.iloc[-1]['ss_id']

        ssmin = self.get_by_name(ssmin_id)
        ssmax = self.get_by_name(ssmax_id)

        return ssmin, ssmax

    @property
    def torsion_array(self):
        return(self.get_torsion_array())
    
    def get_torsion_array(self):
        """
        Returns an rows X 5 array representing the dihedral angles
        in the given disulfide list.

        """
        sslist = self.data
        tot = len(sslist)
        res = numpy.zeros(shape=(tot, 5))

        for idx, ss in zip(range(tot), sslist):
            row = ss.torsion_array
            res[idx] = row
        return res

    def Torsion_RMS(self):
        '''
        Calculate the RMS distance in torsion space between all pairs in the
        DisulfideList
        '''
        sslist = self.data
        tot = len(sslist)

        totsq = 0.0
        for ss1 in sslist:
            tors1 = ss1.torsion_array
            total1 = 0
            for ss2 in sslist:
                tors2 = ss2.torsion_array
                total1 += proteusPy.dist_squared(tors1, tors2)
        
            totsq = totsq + (total1 / tot)

        return(math.sqrt(totsq/tot**2))

    # Rendering engine calculates and instantiates all bond 
    # cylinders and atomic sphere meshes. Called by all high level routines

    def _render(self, style) -> pv.Plotter:
        ''' 
            Display a window showing the list of disulfides in the given style.
            Argument:
                self
                style: one of 'cpk', 'bs', 'sb', 'plain', 'cov', 'pd'
            Returns:
                Window displaying the Disulfides.
        '''
        
        ssList = self.data
        tot_ss = len(ssList) # number off ssbonds
        rows, cols = grid_dimensions(tot_ss)
        winsize = (512 * cols, 512 * rows)

        pl = pv.Plotter(window_size=winsize, shape=(rows, cols))

        i = 0

        for r in range(rows):
            for c in range(cols):
                pl.subplot(r,c)
                if i < tot_ss:
                    ss = ssList[i]
                    src = ss.pdb_id
                    enrg = ss.energy
                    title = f'{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol Ca: {ss.ca_distance:.2f}'
                    pl.add_title(title=title, font_size=FONTSIZE)
                    ss._render(pl, style=style, bondcolor=BOND_COLOR, bs_scale=BS_SCALE, 
                            spec=SPECULARITY, specpow=SPEC_POWER)
                i += 1
        return pl

    def display(self, style='sb'):
        pl = pv.Plotter()

        pl = self._render(style)
        pl.add_camera_orientation_widget()
        pl.enable_anti_aliasing('msaa')
        pl.link_views()
        pl.reset_camera()
        pl.show()
        
    def screenshot(self, style='bs', fname='sslist.png', verbose=True):
        ''' 
            Save the interactive window displaying the list of disulfides in the given style.
            Argument:
                self
                style: one of 'cpk', 'bs', 'sb', 'plain', 'cov', 'pd'
                fname: filename for the resulting image file.
            Returns:
                Image file saved to disk.
        '''

        pl = pv.Plotter()
        pl = self._render(style=style)

        pl.enable_anti_aliasing('msaa')
        pl.link_views()
        pl.reset_camera()
        pl.show(auto_close=False)

        if verbose:
            print(f'Saving file: {fname}')
        
        pl.screenshot(fname)
        
        if verbose:
            print(f'Saved file: {fname}')
        
        return
    
    def display_overlay(self, screenshot=False, movie=False, 
                        verbose=True,
                        fname='ss_overlay.png'):
        ''' 
        Display all disulfides in the list overlaid in stick mode against
        a common coordinate frames. This allows us to see all of the disulfides
        at one time in a single view. Colors vary smoothy between bonds.
        
        Arguments:
            PDB_SS: DisulfideLoader object initialized with the database.
            pdbid: the actual PDB id string

        Returns: None.    
        ''' 
        id = self.pdb_id
        ssbonds = self.data
        tot_ss = len(ssbonds) # number off ssbonds

        tot_ss = len(ssbonds) # number off ssbonds
        title = f'Disulfides for SS list {id}: ({tot_ss} total)'

        if movie:
            pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
        else:
            pl = pv.Plotter(window_size=WINSIZE, off_screen=False)

        pl.add_title(title=title, font_size=FONTSIZE)
        pl.enable_anti_aliasing('msaa')
        pl.add_axes()

        mycol = numpy.zeros(shape=(tot_ss, 3))
        mycol = proteusPy.cmap_vector(tot_ss)

        for i, ss in zip(range(tot_ss), ssbonds):
            color = [int(mycol[i][0]), int(mycol[i][1]), int(mycol[i][2])]
            ss._render(pl, style='plain', bondcolor=color, translate=False)

        pl.reset_camera()

        if screenshot:
            pl.show(auto_close=False) # allows for manipulation
            pl.screenshot(fname)
        elif movie:
            if verbose:
                print(f'Saving mp4 animation to: {fname}')
                
            pl.open_movie(fname, quality=9)
            path = pl.generate_orbital_path(n_points=360)
            pl.orbit_on_path(path, write_frames=True)
            pl.close()

            if verbose:
                print(f'Saved mp4 animation to: {fname}')
        else:
            pl.show()
        
        return

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# end of file

        
