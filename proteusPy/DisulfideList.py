# DisulfideList Class definition
# I extend UserList to handle lists of Disulfide objects.
# Indexing and slicing are supported, sorting is based on energy
# Author Eric G. Suchanek, PhD
# A part of the proteusPy molecular modeling and analysis suite by
# Eric G. Suchanek, PhD
# Last modification: 12/13/2023 -egs-

import proteusPy
from proteusPy import *

import pyvista as pv
from collections import UserList

def grid_dimensions(n):
    '''
    Calculate rows and columns for the given needed to display
    a given number of disulfides in a square aspect.

    Arguments: n
    Returns: rows, columns
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


class DisulfideList(UserList):
    '''
    Class provides a sortable list for Disulfide objects.
    Indexing and slicing are supported, and normal list operations like .insert, .append and .extend.
    The DisulfideList object must be initialized with an iterable (tuple, list) and a name.
    
    The class can also render Disulfides to a pyVista window using the DisulfideList.display() 
    method. See below for examples.\n

    Examples:
        from proteusPy.Disulfide import DisulfideList, Disulfide, DisulfideLoader
        
        # instantiate some variables
        SS = Disulfide()
        # Note: the list is initialized with an iterable and a name (optional)
        SSlist = DisulfideList([],'ss')

        PDB_SS = DisulfideLoader()  # load the Disulfide database\n
        SS = PDB_SS[0]              # returns a Disulfide object at index 0
        SSlist = PDB_SS['4yys']     # returns a DisulfideList containing all
                                    #  disulfides for 4yys\n

        SSlist = PDB_SS[:8]         # get SS bonds for the last 8 structures\n
        SSlist.display('sb')        # render the disulfides in 'split bonds' style\n

        # make some empty disulfidesx
        ss1 = Disulfide('ss1')
        ss2 = Disulfide('ss2')

        # make a DisulfideList containing ss1, named 'tmp'
        sslist = DisulfideList([ss1], 'tmp')
        sslist.append(ss2)

        # extract the first disulfide
        ss1 = PDB_SS[0]
        print(f'{ss1.pprint_all()}')

        # grab a list of disulfides via slicing
        subset = DisulfideList(PDB_SS[0:10],'subset')
        subset.display(style='sb')      # display the disulfides in 'split bond' style
        subset.display_overlay()        # display all disulfides overlaid in stick style
        subset.screenshot(style='sb', fname='subset.png')  # save a screenshot.
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
    
    def set_id(self, value):
        self.pdb_id = value
    
    def get_id(self):
        return self.pdb_id

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
        return res

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
                print(f'Cross chain SS: {ss.print_compact}:')
        return reslist
    
    def _render(self, style='sb') -> pv.Plotter:
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
        i = 0

        WINSIZE = (512 * cols, 512 * rows)
        pl = pv.Plotter(window_size=WINSIZE, shape=(rows, cols))
        pl.add_camera_orientation_widget()

        for r in range(rows):
            for c in range(cols):
                pl.subplot(r,c)
                if i < tot_ss:
                    pl.enable_anti_aliasing('msaa')
                    pl.view_isometric()
                    ss = ssList[i]
                    src = ss.pdb_id
                    enrg = ss.energy
                    near_range, far_range = ss.compute_extents()
                    title = f'{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol'
                    pl.add_title(title=title, font_size=FONTSIZE)
                    ss._render(pl, style=style, bondcolor=BOND_COLOR, 
                                   bs_scale=BS_SCALE, spec=SPECULARITY, specpow=SPEC_POWER)
                    pl.view_isometric()
                i += 1
        
        pl.link_views()
        pl.reset_camera()
        return pl

    def display(self, style='sb'):
        pl = pv.Plotter()
        pl = self._render(style)
        pl.show()


        #pl.show()
    
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
        pl.show(auto_close=False)

        if verbose:
            print(f'Saving file: {fname}')
        
        pl.screenshot(fname)
        
        if verbose:
            print(f'Saved file: {fname}')
        
        
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

        i = 0
        for ss in ssbonds:
            color = [int(mycol[i][0]), int(mycol[i][1]), int(mycol[i][2])]
            ss._render(pl, style='plain', bondcolor=color, translate=False)
            i += 1

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


# end of file

        
