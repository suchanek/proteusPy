# Implementation for a Disulfide Bond structural object.
# Based on the original C/C++ implementation by Eric G. Suchanek,
# A part of the program Proteus, a program for the analysis and modeling of 
# protein structures, with an emphasis on disulfide bonds.
# Author: Eric G. Suchanek, PhD
# Last revision: 1/24/2023
# Cα N, Cα, Cβ, C', Sγ Å °

import math
import numpy
import copy

from math import cos

import pickle
from tqdm import tqdm

import pandas as pd
import pyvista as pv

import proteusPy
from proteusPy import *
from proteusPy.atoms import *

from proteusPy.data import SS_DICT_PICKLE_FILE, SS_ID_FILE
from proteusPy.data import SS_PICKLE_FILE, SS_TORSIONS_FILE, PROBLEM_ID_FILE

from proteusPy.DisulfideExceptions import *
from proteusPy.DisulfideList import DisulfideList

from Bio.PDB import Vector, PDBParser, PDBList
from Bio.PDB.vectors import calc_dihedral

# float init for class 
_FLOAT_INIT = -999.9
_ANG_INIT = -180.0

# tqdm progress bar width
_PBAR_COLS = 100
WINSIZE = (1024, 1024)

# columns for the torsions file dataframe.
global Torsion_DF_Cols

Torsion_DF_Cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', \
           'chi5', 'energy', 'ca_distance', 'phi_prox', 'psi_prox', 'phi_dist',\
           'psi_dist', 'torsion_length']


# Class definition for a Disulfide bond. 
class Disulfide:
    """
    This class provides a Python object and methods representing a physical disulfide bond 
    either extracted from the RCSB protein databank or built using the 
    [proteusPy.Turtle3D](turtle3D.html) class. The disulfide bond is characterized 
    by:
    * Atomic coordinates for the atoms N, Cα, Cβ, C', Sγ for both residues. 
    These are stored as both raw atomic coordinates as read from the RCSB file 
    and internal local coordinates.
    * The dihedral angles Χ1 - Χ5 for the disulfide bond
    * A name, by default {pdb_id}{prox_resnumb}{prox_chain}_{distal_resnum}{distal_chain} 
    * Proximal residue number
    * Distal residue number
    * Approximate torsional energy (kcal/mol):
    
    $$ 
    E_{kcal/mol} \\approx 2.0 * cos(3.0 * \\chi_{1}) + cos(3.0 * \\chi_{5}) + cos(3.0 * \\chi_{2}) + 
    $$

    $$
    cos(3.0 * \\chi_{4}) + 3.5 * cos(2.0 * \\chi_{3}) + 0.6 * cos(3.0 * \\chi_{3}) + 10.1 
    $$
    
    The equation embodies the typical 3-fold rotation barriers associated with single bonds,
    (Χ1, Χ5, Χ2, Χ4) and a high 2-fold barrier for Χ3, resulting from the partial double bond
    character of the S-S bond. This property leads to two major disulfide families, characterized
    by the sign of Χ3. *Left-handed* disulfides have Χ3 < 0° and *right-handed* disulfides have
    Χ3 > 0°.

    * Euclidean length of the dihedral angles (degrees) defined as:
    $$\\sqrt(\\sum \\chi_{1}^{2} + \\chi_{2}^{2} + \\chi_{3}^{2} + \\chi_{4}^{2} + \\chi_{5}^{2})$$
    * Cα - Cα distance (Å)
    * The previous C' and next N for both the proximal and distal residues. These are needed
    to calculate the backbone dihedral angles Φ and Ψ.
    * Backbone dihedral angles Φ and Ψ, when possible. Not all structures are complete and
    in those cases the atoms needed may be undefined. In this case the Φ and Ψ angles are set
    to -180°.
    
    The class also provides a rendering capabilities using the excellent [PyVista](https://pyvista.org)
    library, and can display disulfides interactively in a variety of display styles:
    * 'sb' - Split Bonds style - bonds colored by their atom type
    * 'bs' - Ball and Stick style - split bond coloring with small atoms
    * 'pd' - Proximal/Distal style - bonds colored *Red* for proximal residue and *Green* for
    the distal residue.
    * 'cpk' - CPK style rendering, colored by atom type:
        * Carbon   - Grey
        * Nitrogen - Blue
        * Sulfur   - Yellow
        * Oxygen   - Red
        * Hydrogen - White 
    
    Individual displays can be saved to a file, and animations created.

    """
    def __init__(self, name: str="SSBOND") -> None:
        '''
        __init__ Initialize the class to defined internal values.

        Parameters
        ----------
        name : str, optional \n
            Disulfide name, by default "SSBOND"
        '''
        self.name = name
        self.proximal = -1
        self.distal = -1
        self.energy = _FLOAT_INIT
        self.proximal_chain = str('')
        self.distal_chain = str('')
        self.pdb_id = str('')
        self.proximal_residue_fullid = str('')
        self.distal_residue_fullid = str('')
        self.PERMISSIVE = bool(True)
        self.QUIET = bool(True)
        self.ca_distance = _FLOAT_INIT
        self.torsion_array = numpy.array((_ANG_INIT, _ANG_INIT, _ANG_INIT, 
        								  _ANG_INIT, _ANG_INIT))
        self.phiprox = _ANG_INIT
        self.psiprox = _ANG_INIT
        self.phidist = _ANG_INIT
        self.psidist = _ANG_INIT

        # global coordinates for the Disulfide, typically as 
        # returned from the PDB file

        self.n_prox = Vector(0,0,0)
        self.ca_prox = Vector(0,0,0)
        self.c_prox = Vector(0,0,0)
        self.o_prox = Vector(0,0,0)
        self.cb_prox = Vector(0,0,0)
        self.sg_prox = Vector(0,0,0)
        self.sg_dist = Vector(0,0,0)
        self.cb_dist = Vector(0,0,0)
        self.ca_dist = Vector(0,0,0)
        self.n_dist = Vector(0,0,0)
        self.c_dist = Vector(0,0,0)
        self.o_dist = Vector(0,0,0)

        # set when we can't find previous or next prox or distal
        # C' or N atoms.
        self.missing_atoms = False

        # need these to calculate backbone dihedral angles
        self.c_prev_prox = Vector(0,0,0)
        self.n_next_prox = Vector(0,0,0)
        self.c_prev_dist = Vector(0,0,0)
        self.n_next_dist = Vector(0,0,0)

        # local coordinates for the Disulfide, computed using the Turtle3D in 
        # Orientation #1. these are generally private.

        self._n_prox = Vector(0,0,0)
        self._ca_prox = Vector(0,0,0)
        self._c_prox = Vector(0,0,0)
        self._o_prox = Vector(0,0,0)
        self._cb_prox = Vector(0,0,0)
        self._sg_prox = Vector(0,0,0)
        self._sg_dist = Vector(0,0,0)
        self._cb_dist = Vector(0,0,0)
        self._ca_dist = Vector(0,0,0)
        self._n_dist = Vector(0,0,0)
        self._c_dist = Vector(0,0,0)
        self._o_dist = Vector(0,0,0)

        # need these to calculate backbone dihedral angles
        self._c_prev_prox = Vector(0,0,0)
        self._n_next_prox = Vector(0,0,0)
        self._c_prev_dist = Vector(0,0,0)
        self._n_next_dist = Vector(0,0,0)

        # Dihedral angles for the disulfide bond itself, set to _ANG_INIT
        self.chi1 = _ANG_INIT
        self.chi2 = _ANG_INIT
        self.chi3 = _ANG_INIT
        self.chi4 = _ANG_INIT
        self.chi5 = _ANG_INIT

        self.torsion_length = _FLOAT_INIT

    def cofmass(self) -> numpy.array:
        '''
        Returns the geometric center of mass for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        Returns
        -------
        numpy.array: \n
            3D array for the geometric center of mass
        '''

        res = self.internal_coords()
        return res.mean(axis=0)

    def internal_coords(self) -> numpy.array:
        '''
        Returns the internal coordinates for the Disulfide.
        If there are missing atoms the extra atoms for the proximal
        and distal N and C are set to [0,0,0]. This is needed for the center of
        mass calculations, used when rendering.
        
        Returns
        -------
        numpy.array: \n
            Array containing the coordinates, [16][3].
        '''
        
        # if we don't have the prior and next atoms we initialize those
        # atoms to the origin so as to not effect the center of mass calculations
        if self.missing_atoms:
            res_array = numpy.array((
                self._n_prox.get_array(),
                self._ca_prox.get_array(),
                self._c_prox.get_array(),
                self._o_prox.get_array(), 
                self._cb_prox.get_array(),
                self._sg_prox.get_array(),
                self._n_dist.get_array(),
                self._ca_dist.get_array(),
                self._c_dist.get_array(), 
                self._o_dist.get_array(), 
                self._cb_dist.get_array(),
                self._sg_dist.get_array(),
                [0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ))
        else:
            res_array = numpy.array((
                self._n_prox.get_array(),
                self._ca_prox.get_array(),
                self._c_prox.get_array(), 
                self._o_prox.get_array(), 
                self._cb_prox.get_array(),
                self._sg_prox.get_array(),
                self._n_dist.get_array(),
                self._ca_dist.get_array(),
                self._c_dist.get_array(), 
                self._o_dist.get_array(), 
                self._cb_dist.get_array(),
                self._sg_dist.get_array(),
                self._c_prev_prox.get_array(),
                self._n_next_prox.get_array(),
                self._c_prev_dist.get_array(),
                self._n_next_dist.get_array()
            ))
        return res_array
    
    def internal_coords_res(self, resnumb) -> numpy.array:
        '''
        Returns the internal coordinates for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        Parameters
        ----------
        resnumb : int \n
            Residue number for disulfide

        Returns
        -------
        numpy.array \n
            Array containing the internal coordinates for the disulfide

        Raises
        ------
        DisulfideConstructionWarning \n
            Warning raised if the residue number is invalid
        '''
        res_array = numpy.zeros(shape=(6,3))

        if resnumb == self.proximal:
            res_array = numpy.array((
                self._n_prox.get_array(),
                self._ca_prox.get_array(),
                self._c_prox.get_array(), 
                self._o_prox.get_array(), 
                self._cb_prox.get_array(),
                self._sg_prox.get_array(),
            ))
            return res_array

        elif resnumb == self.distal:
            res_array = numpy.array((
                self._n_dist.get_array(),
                self._ca_dist.get_array(),
                self._c_dist.get_array(),
                self._o_dist.get_array(), 
                self._cb_dist.get_array(),
                self._sg_dist.get_array(),
            ))
            return res_array
        else:
            mess = f'-> Disulfide.internal_coords(): Invalid argument. \
             Unable to find residue: {resnumb} '
            raise DisulfideConstructionWarning(mess)

    def get_chains(self) -> tuple:
        '''
        Return the proximal and distal chain IDs for the Disulfide.

        Returns
        -------
        tuple \n
            (proximal, distal) chain IDs
        '''
        prox = self.proximal_chain
        dist = self.distal_chain
        return tuple(prox, dist)
    
    def same_chains(self) -> bool:
        '''
        Function checks if the Disulfide is cross-chain or not.

        Returns
        -------
        bool \n
            True if the proximal and distal residues are on the same chains,
            False otherwise.
        '''

        (prox, dist) = self.get_chains()
        return prox == dist
        
    def reset(self) -> None:
        '''
        Resets the disulfide object to its initial state. All distances, 
        angles and positions are reset. The name is unchanged.
        '''
        self.__init__(self)

    def copy(self):
        '''
        Copy the Disulfide

        Returns
        -------
        Disulfide \n
            A copy of self.
        '''
        
        return copy.deepcopy(self)
    
    def compute_extents(self, dim='z'):
        '''
        Calculate the internal coordinate extents for the input axis.

        Parameters
        ----------
        dim : str, optional
            Axis, one of 'x', 'y', 'z', by default 'z'

        Returns
        -------
        float \n
            min, max
        '''
        ic = self.internal_coords()
        # set default index to 'z'
        idx = 2

        if dim == 'x':
            idx = 0
        elif dim == 'y':
            idx = 1
        elif dim == 'z':
            idx = 2
        
        _min = ic[:, idx].min()
        _max = ic[:, idx].max()
        return _min, _max

    def bounding_box(self):
        '''
        Return the bounding box array for the given disulfide

        Returns
        -------
        numpy.Array(3,2)

            Array containing the min, max for X, Y, and Z respectively. 
            Does not currently take the atom's radius into account.
        '''
        res = numpy.zeros(shape=(3, 2))
        xmin, xmax = self.compute_extents('x')
        ymin, ymax = self.compute_extents('y')
        zmin, zmax = self.compute_extents('z')

        res[0] = [xmin, xmax]
        res[1] = [ymin, ymax]
        res[2] = [zmin, zmax]

        return res

    def _render(self, pvplot: pv.Plotter, style='bs', plain=False,
            bondcolor=BOND_COLOR, bs_scale=BS_SCALE, spec=SPECULARITY, 
            specpow=SPEC_POWER, translate=True):
        '''
        Update the passed pyVista plotter() object with the mesh data for the
        input Disulfide Bond. Used internally

        Parameters
        ----------
        pvplot : pv.Plotter
            pyvista.Plotter object

        style : str, optional
            Rendering style, by default 'bs'. One of 'bs', 'st', 'cpk', Render as \
            CPK, ball-and-stick or stick. Bonds are colored by atom color, unless \
            'plain' is specified.

        plain : bool, optional
            Used internally, by default False

        bondcolor : pyVista color name, optional bond color for simple bonds, by default BOND_COLOR

        bs_scale : float, optional
            scale factor (0-1) to reduce the atom sizes for ball and stick, by default BS_SCALE
        spec : float, optional
            specularity (0-1), where 1 is totally smooth and 0 is rough, by default SPECULARITY

        specpow : int, optional
            exponent used for specularity calculations, by default SPEC_POWER

        translate : bool, optional
            Flag used internally to indicate if we should translate \
            the disulfide to its geometric center of mass, by default True.

        Returns
        -------
        pv.Plotter
            Updated pv.Plotter object with atoms and bonds.
        '''
        
        _bradius = BOND_RADIUS
        coords = self.internal_coords()
        missing_atoms = self.missing_atoms
        clen = coords.shape[0]
        
        if translate:
            cofmass = self.cofmass()
            for i in range(clen):
                coords[i] = coords[i] - cofmass
        
        atoms = ('N', 'C', 'C', 'O', 'C', 'SG', 'N', 'C', 'C', 'O', 'C',
        		 'SG', 'C', 'N', 'C', 'N')
        pvp = pvplot
        
        # bond connection table with atoms in the specific order shown above: 
        # returned by ss.get_internal_coords()
        
        def draw_bonds(pvp, bradius=BOND_RADIUS, style='sb', 
        			   bcolor=BOND_COLOR, missing=True):
            '''
            Generate the appropriate pyVista cylinder objects to represent
            a particular disulfide bond. This utilizes a connection table 
            for the starting and ending atoms and a color table for the 
            bond colors. Used internally.

            Parameters
            ----------
            pvp: pyVista.Plotter 
                input plotter object to be updated

            bradius: float
                bond radius

            style: str
                bond style. One of sb, plain, pd

            bcolor: pyvista color

            missing: bool
                True if atoms are missing, False othersie
            
            Returns
            -------
            pvp: pyvista.Plotter
                Updated Plotter object.
            '''
            bond_conn = numpy.array(
            [
                [0, 1], # n-ca
                [1, 2], # ca-c
                [2, 3], # c-o
                [1, 4], # ca-cb
                [4, 5], # cb-sg
                [6, 7], # n-ca
                [7, 8], # ca-c
                [8, 9], # c-o
                [7, 10], # ca-cb
                [10, 11], # cb-sg
                [5, 11],  # sg -sg
                [12, 0],  # cprev_prox-n
                [2, 13],  # c-nnext_prox
                [14,6],   # cprev_dist-n_dist
                [8,15]    # c-nnext_dist
            ])
            
            # colors for the bonds. Index into ATOM_COLORS array
            bond_split_colors = numpy.array(
                [
                    ('N', 'C'),
                    ('C', 'C'),
                    ('C', 'O'),
                    ('C', 'C'),
                    ('C', 'SG'),
                    ('N', 'C'),
                    ('C', 'C'),
                    ('C', 'O'),
                    ('C', 'C'),
                    ('C', 'SG'),
                    ('SG', 'SG'),
                    # prev and next C-N bonds - color by atom Z
                    ('C', 'N'),
                    ('C', 'N'),
                    ('C', 'N'),
                    ('C', 'N')
                ]
            )
            # work through connectivity and colors
            orig_col = dest_col = bcolor

            for i in range(len(bond_conn)):
                if i > 10 and missing_atoms == True: # skip missing atoms
                    continue

                bond = bond_conn[i]

                # get the indices for the origin and destination atoms
                orig = bond[0]
                dest = bond[1]

                col = bond_split_colors[i]

                # get the coords
                prox_pos = coords[orig]
                distal_pos = coords[dest]
                
                # compute a direction vector
                direction = distal_pos - prox_pos

                # compute vector length. divide by 2 since split bond
                height = math.dist(prox_pos, distal_pos) / 2.0

				# the cylinder origins are actually in the 
				# middle so we translate
                
                origin = prox_pos + 0.5 * direction # for a single plain bond
                origin1 = prox_pos + 0.25 * direction 
                origin2 = prox_pos + 0.75 * direction
                
                bradius = _bradius

                if style == 'plain':
                    orig_col = dest_col = bcolor
                
                # proximal-distal red/green coloring
                elif style == 'pd':
                    if i <= 4 or i == 11 or i == 12:
                        orig_col = dest_col = 'red'
                    else:
                        orig_col = dest_col= 'green'
                    if i == 10:
                        orig_col = dest_col= 'yellow'
                else:
                    orig_col = ATOM_COLORS[col[0]]
                    dest_col = ATOM_COLORS[col[1]]

                if i >= 11: # prev and next residue atoms for phi/psi calcs
                    bradius = _bradius * .5 # make smaller to distinguish
                
                cap1 = pv.Sphere(center=prox_pos, radius=bradius)
                cap2 = pv.Sphere(center=distal_pos, radius=bradius)

                if style == 'plain':
                    cyl = pv.Cylinder(origin, direction, radius=bradius, height=height*2.0) 
                    pvp.add_mesh(cyl, color=orig_col)
                else:
                    cyl1 = pv.Cylinder(origin1, direction, radius=bradius, height=height)
                    cyl2 = pv.Cylinder(origin2, direction, radius=bradius, height=height)
                    pvp.add_mesh(cyl1, color=orig_col)
                    pvp.add_mesh(cyl2, color=dest_col)
        
                pvp.add_mesh(cap1, color=orig_col)
                pvp.add_mesh(cap2, color=dest_col)

            return pvp # end draw_bonds
        
        if style=='cpk':
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_CPK[atom]
                pvp.add_mesh(pv.Sphere(center=coords[i], radius=rad), color=ATOM_COLORS[atom], 
                             smooth_shading=True, specular=spec, specular_power=specpow)
                i += 1
        
        elif style=='cov':
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_COVALENT[atom]
                pvp.add_mesh(pv.Sphere(center=coords[i], radius=rad), color=ATOM_COLORS[atom], 
                            smooth_shading=True, specular=spec, specular_power=specpow)
                i += 1

        elif style == 'bs': # ball and stick
            i = 0
            for atom in atoms:
                rad = ATOM_RADII_CPK[atom] * bs_scale
                if i > 11:
                    rad = rad * .75
                
                pvp.add_mesh(pv.Sphere(center=coords[i], radius=rad), 
                			 color=ATOM_COLORS[atom], smooth_shading=True, 
                			 specular=spec, specular_power=specpow)
                i += 1
            pvp = draw_bonds(pvp, style='bs')

        elif style == 'sb': # splitbonds
            pvp = draw_bonds(pvp, style='sb', missing=missing_atoms)
        
        elif style == 'pd': # proximal-distal
            pvp = draw_bonds(pvp, style='pd', missing=missing_atoms)

        else: # plain
            pvp = draw_bonds(pvp, style='plain', bcolor=bondcolor, 
            				missing=missing_atoms)
            
        return

    def display(self, single=True, style='sb'):
        '''
        Display the Disulfide bond in the specific rendering style.

        Parameters
        ----------
        single: bool 
            Display the bond in a single panel in the specific style.

        style: str
            One of: \n
            'sb' - split bonds
            'bs' - ball and stick
            'cpk' - CPK style
            'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            'plain' - boring single color

        Example:
        >>> import proteusPy
        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideLoader import DisulfideLoader

        >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)
        >>> ss = PDB_SS[0]
        >>> ss.display(style='cpk')
        
        ss.screenshot(style='cpk', fname='proteus_logo2.png')
        '''
        src = self.pdb_id
        enrg = self.energy
        title = f'{src}: {self.proximal}{self.proximal_chain}-{self.distal}{self.distal_chain}: {enrg:.2f} kcal/mol. Ca: {self.ca_distance:.2f} Å'
                
        if single == True:
            _pl = pv.Plotter(window_size=WINSIZE)
            _pl.add_title(title=title, font_size=FONTSIZE)
            _pl.enable_anti_aliasing('msaa')
            _pl.add_camera_orientation_widget()            

            self._render(_pl, style=style, bs_scale=BS_SCALE, 
                        spec=SPECULARITY, specpow=SPEC_POWER)        
            _pl.reset_camera()
            _pl.show()

        else:
            pl = pv.Plotter(window_size=WINSIZE, shape=(2,2))
            pl.subplot(0,0)
            
            pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing('msaa')

            pl.add_camera_orientation_widget()
            
            self._render(pl, style='cpk', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.subplot(0,1)
            pl.add_title(title=title, font_size=FONTSIZE)
            
            self._render(pl, style='bs', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.subplot(1,0)
            pl.add_title(title=title, font_size=FONTSIZE)
            
            self._render(pl, style='sb', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.subplot(1,1)
            pl.add_title(title=title, font_size=FONTSIZE)
            
            self._render(pl, style='pd', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.link_views()
            pl.reset_camera()
            pl.show()
        return
    
    def screenshot(self, single=True, style='sb', fname='ssbond.png', verbose=False):
        '''
        Create and save a screenshot of the Disulfide in the given style
        and filename

        Parameters
        ----------
        single : bool, optional
            Display a single vs panel view, by default True

        style : str, optional \n
            One of: \n
            'sb' - split bonds
            'bs' - ball and stick
            'cpk' - CPK style
            'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            'plain' - boring single color, by default 'sb'

        fname : str, optional
            output filename, by default 'ssbond.png'

        verbose : bool, optional
            Verbosity, by default False.
        '''
        src = self.pdb_id
        ssname = self.name
        enrg = self.energy
        title = f'{src}: {ssname}: {enrg:.2f} kcal/mol'
        
        if verbose:
            print(f'Rendering screenshot...')

        if single:
            pl = pv.Plotter(window_size=WINSIZE)
            pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing('msaa')
            pl.add_camera_orientation_widget()
            self._render(pl, style=style, bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, specpow=SPEC_POWER)
            pl.reset_camera()
            pl.show(auto_close=False)
            pl.screenshot(fname)
            pl.clear()

        else:
            pl = pv.Plotter(window_size=WINSIZE, shape=(2,2))
            pl.subplot(0,0)
            
            pl.add_title(title=title, font_size=FONTSIZE)
            pl.enable_anti_aliasing('msaa')

            pl.add_camera_orientation_widget()
            self._render(pl, style='cpk', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.subplot(0,1)
            pl.add_title(title=title, font_size=FONTSIZE)
            self._render(pl, style='pd', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.subplot(1,0)
            pl.add_title(title=title, font_size=FONTSIZE)
            self._render(pl, style='bs', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)
            
            pl.subplot(1,1)
            pl.add_title(title=title, font_size=FONTSIZE)
            self._render(pl, style='sb', bondcolor=BOND_COLOR, 
                        bs_scale=BS_SCALE, spec=SPECULARITY, 
                        specpow=SPEC_POWER)

            pl.link_views()
            pl.reset_camera()
            pl.show(auto_close=False)
            pl.screenshot(fname)
        
        if verbose:
            print(f'Saved: {fname}')

    def make_movie(self, style='sb', fname='ssbond.mp4',
                   verbose=False, steps=360):
        '''
        Create and save an animation for the given Disulfide in the 
        given style and filename.

        Parameters
        ----------
        style : str, optional

            One of 
            'sb' - split bonds
            'bs' - ball and stick
            'cpk' - CPK style
            'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            'plain' - boring single color, by default 'sb'

        fname : str, optional
            output filename, by default 'ssbond.png'
        verbose : bool, optional
            Verbosity, by default False.
        steps : int
            Number of steps for the rotation, by default 360.
        '''
        src = self.pdb_id
        ssname = self.name
        enrg = self.energy
        title = f'{src}: {ssname}: {enrg:.2f} kcal/mol'
        
        if verbose:
            print(f'Rendering animation to {fname}...')

        pl = pv.Plotter(window_size=WINSIZE, off_screen=True)
        pl.open_movie(fname, quality=9)
        path = pl.generate_orbital_path(n_points=steps)

        pl.add_title(title=title, font_size=FONTSIZE)
        pl.enable_anti_aliasing('msaa')
        pl.add_camera_orientation_widget()
        pl = self._render(pl, style=style, bondcolor=BOND_COLOR, 
                    bs_scale=BS_SCALE, spec=SPECULARITY, specpow=SPEC_POWER)
        pl.reset_camera()
        pl.orbit_on_path(path, write_frames=True)
        pl.close()

        if verbose:
            print(f'Saved mp4 animation to: {fname}')
        
    # comparison operators, used for sorting. keyed to SS bond energy
    def __lt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy < other.energy

    def __le__(self, other):
        if isinstance(other, Disulfide):
            return self.energy <= other.energy
    
    def __gt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy > other.energy

    def __ge__(self, other):
        if isinstance(other, Disulfide):
            return self.energy >= other.energy

    def __eq__(self, other):
        if isinstance(other, Disulfide):
            return self.energy == other.energy

    def __ne__(self, other):
        if isinstance(other, Disulfide):
            return self.energy != other.energy
    
    # repr functions. The class is large, so I split it up into sections
    def repr_ss_info(self):
        """
        Representation for the Disulfide class
        """
        s1 = f'<Disulfide {self.name} SourceID: {self.pdb_id} Proximal: {self.proximal} {self.proximal_chain} Distal: {self.distal} {self.distal_chain}'
        return s1
    
    def repr_ss_coords(self):
        """
        Representation for Disulfide coordinates
        """
        s2 = f'\nProximal Coordinates:\n   N: {self.n_prox}\n   Cα: {self.ca_prox}\n   C: {self.c_prox}\n   O: {self.o_prox}\n   Cβ: {self.cb_prox}\n   Sγ: {self.sg_prox}\n   Cprev {self.c_prev_prox}\n   Nnext: {self.n_next_prox}\n'
        s3 = f'Distal Coordinates:\n   N: {self.n_dist}\n   Cα: {self.ca_dist}\n   C: {self.c_dist}\n   O: {self.o_dist}\n   Cβ: {self.cb_dist}\n   Sγ: {self.sg_dist}\n   Cprev {self.c_prev_dist}\n   Nnext: {self.n_next_dist}\n\n'
        stot = f'{s2} {s3}'
        return stot

    def repr_ss_conformation(self):
        """
        Representation for Disulfide conformation
        """
        s4 = f'Conformation: (Χ1-Χ5):  {self.chi1:.3f}°, {self.chi2:.3f}°, {self.chi3:.3f}°, {self.chi4:.3f}° {self.chi5:.3f}° '
        s5 = f'Energy: {self.energy:.3f} kcal/mol'
        stot = f'{s4} {s5}'
        return stot

    def repr_ss_local_coords(self):
        """
        Representation for the Disulfide internal coordinates.
        """
        s2i = f'Proximal Internal Coords:\n   N: {self._n_prox}\n   Cα: {self._ca_prox}\n   C: {self._c_prox}\n   O: {self._o_prox}\n   Cβ: {self._cb_prox}\n   Sγ: {self._sg_prox}\n   Cprev {self.c_prev_prox}\n   Nnext: {self.n_next_prox}\n'
        s3i = f'Distal Internal Coords:\n   N: {self._n_dist}\n   Cα: {self._ca_dist}\n   C: {self._c_dist}\n   O: {self._o_dist}\n   Cβ: {self._cb_dist}\n   Sγ: {self._sg_dist}\n   Cprev {self.c_prev_dist}\n   Nnext: {self.n_next_dist}\n'
        stot = f'{s2i}{s3i}'
        return stot
    
    def repr_ss_chain_ids(self):
        """
        Representation for Disulfide chain IDs
        """
        return(f'Proximal Chain fullID: <{self.proximal_residue_fullid}> Distal Chain fullID: <{self.distal_residue_fullid}>')

    def repr_ss_ca_dist(self):
        """
        Representation for Disulfide Ca distance
        """
        s1 = f'Cα Distance: {self.ca_distance:.3f} Å'
        return s1
    
    def repr_ss_torsion_length(self):
        """
        Representation for Disulfide torsion length
        """
        s1 = f'Torsion length: {self.torsion_length:.3f} deg'
        return s1
    
    def __repr__(self):
        """
        Representation for the Disulfide class
        """
        
        s1 = self.repr_ss_info()
        res = f'{s1}>'
        return res

    def pprint(self) -> None:
        """
        Pretty print general info for the Disulfide
        """
        
        s1 = self.repr_ss_info()
        s2 = self.repr_ss_ca_dist()
        s3 = self.repr_ss_conformation()
        s4 = self.repr_ss_torsion_length()
        res = f'{s1} \n{s3} \n{s2} \n{s4}>'
        print(res)

    def repr_all(self) -> str:
        """
        Return a string representation for all Disulfide information
        contained in self.
        """
        
        s1 = self.repr_ss_info() + '\n'
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s5 = self.repr_chain_ids()
        s6 = self.repr_ss_ca_dist()
        s7 = self.repr_ss_torsion_length()

        res = f'{s1} {s5} {s2} {s3} {s4} {s6} {s7}>'
        return res

    def pprint_all(self) -> None:
        s1 = self.repr_ss_info() + '\n'
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s5 = self.repr_chain_ids()
        s6 = self.repr_ss_ca_dist()
        s7 = self.repr_ss_torsion_length()

        res = f'{s1} {s5} {s2} {s3} {s4}\n {s6}\n {s7}>'

        print(res)
    
    def _handle_SS_exception(self, message: str):
        '''
        This method catches an exception that occurs in the Disulfide
        object (if PERMISSIVE), or raises it again, this time adding the
        PDB line number to the error message. (private).

        Parameters
        ----------
        message : str
            Error message

        Raises
        ------
        DisulfideConstructionException
            Fatal construction exception.
        '''
        # message = "%s at line %i." % (message)
        message = f'{message}'

        if self.PERMISSIVE:
            # just print a warning - some residues/atoms may be missing
            warnings.warn(
                "DisulfideConstructionException: %s\n"
                "Exception ignored.\n"
                "Some atoms may be missing in the data structure."
                % message,
                DisulfideConstructionWarning,
            )
        else:
            # exceptions are fatal - raise again with new message (including line nr)
            raise DisulfideConstructionException(message) from None

    def repr_compact(self) -> str:
        '''
        Return a compact representation of the Disulfide object
        :return: string
        '''
        return(f'{self.repr_ss_info()} {self.repr_ss_conformation()}')

    def repr_conformation(self):
        '''
        Return a string representation of the Disulfide object's conformation.
        :return: string
        '''
        return(f'{self.repr_ss_conformation()}')
    
    def repr_coords(self):
        '''
        Return a string representation of the Disulfide object's coordinates.
        :return: string
        '''
        return(f'{self.repr_ss_coords()}')

    def repr_internal_coords(self):
        '''
        Return a string representation of the Disulfide object's internal coordinaes.
        :return: string
        '''
        return(f'{self.repr_ss_local_coords()}')

    def repr_chain_ids(self):
        '''
        Return a string representation of the Disulfide object's chain ids.
        :return: string
        '''
        return(f'{self.repr_ss_chain_ids()}')

    def set_permissive(self, perm: bool) -> None:
        '''
        Sets PERMISSIVE flag for Disulfide parsing
        :return: None
        '''
        self.PERMISSIVE = perm
    
    def get_permissive(self) -> bool:
        return self.PERMISIVE

    def get_full_id(self):
        return((self.proximal_residue_fullid, self.distal_residue_fullid))
    
    def initialize_disulfide_from_chain(self, chain1, chain2, proximal, 
    									distal, quiet=True):
        '''
        Initialize a new Disulfide object with atomic coordinates from the proximal and 
        distal coordinates, typically taken from a PDB file.

        Arguments: 
            chain1: list of Residues in the model, eg: chain = model['A']
            chain2: list of Residues in the model, eg: chain = model['A']
            proximal: proximal residue sequence ID
            distal: distal residue sequence ID
        
        Returns: none. The internal state is modified.
        '''

        id = chain1.get_full_id()[0]
        self.pdb_id = id
        
        chi1 = chi2 = chi3 = chi4 = chi5 = _ANG_INIT

        prox = int(proximal)
        dist = int(distal)

        prox_residue = chain1[prox]
        dist_residue = chain2[dist]

        if (prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS'):
            print(f'build_disulfide() requires CYS at both residues: {prox} {prox_residue.get_resname()} {dist} {dist_residue.get_resname()} Chain: {prox_residue.get_segid()}')

        # set the objects proximal and distal values
        self.set_resnum(proximal, distal)

        self.proximal_chain = chain1.get_id()
        self.distal_chain = chain2.get_id()

        self.proximal_residue_fullid = prox_residue.get_full_id()
        self.distal_residue_fullid = dist_residue.get_full_id()

        if quiet:
            warnings.filterwarnings("ignore", category=DisulfideConstructionWarning)
        else:
            warnings.simplefilter("always")

        # grab the coordinates for the proximal and distal residues as vectors 
        # so we can do math on them later
        # proximal residue
        
        try:
            n1 = prox_residue['N'].get_vector()
            ca1 = prox_residue['CA'].get_vector()
            c1 = prox_residue['C'].get_vector()
            o1 = prox_residue['O'].get_vector()
            cb1 = prox_residue['CB'].get_vector()
            sg1 = prox_residue['SG'].get_vector()
            
        except Exception:
            raise DisulfideConstructionWarning(f"Invalid or missing coordinates for proximal residue {proximal}") from None
        
        # distal residue
        try:
            n2 = dist_residue['N'].get_vector()
            ca2 = dist_residue['CA'].get_vector()
            c2 = dist_residue['C'].get_vector()
            o2 = dist_residue['O'].get_vector()
            cb2 = dist_residue['CB'].get_vector()
            sg2 = dist_residue['SG'].get_vector()

        except Exception:
            raise DisulfideConstructionWarning(f"Invalid or missing coordinates for distal residue {distal}") from None
        
        # previous residue and next residue - optional, used for phi, psi calculations
        try:
            prevprox = chain1[prox-1]
            nextprox = chain1[prox+1]

            prevdist = chain2[dist-1]
            nextdist = chain2[dist+1]

            cprev_prox = prevprox['C'].get_vector()
            nnext_prox = nextprox['N'].get_vector()

            cprev_dist = prevdist['C'].get_vector()
            nnext_dist = nextdist['N'].get_vector()

            # compute phi, psi for prox and distal
            self.phiprox = numpy.degrees(calc_dihedral(cprev_prox, n1, ca1, c1))
            self.psiprox = numpy.degrees(calc_dihedral(n1, ca1, c1, nnext_prox))
            self.phidist = numpy.degrees(calc_dihedral(cprev_dist, n2, ca2, c2))
            self.psidist = numpy.degrees(calc_dihedral(n2, ca2, c2, nnext_dist))

        except Exception:
            mess = f'Missing coords for: {id} {prox-1} or {dist+1} for SS {proximal}-{distal}'
            cprev_prox = nnext_prox = cprev_dist = nnext_dist = Vector(-1.0, -1.0, -1.0)
            self.missing_atoms = True
            warnings.warn(mess, DisulfideConstructionWarning)

        # update the positions and conformation
        self.set_positions(n1, ca1, c1, o1, cb1, sg1, n2, ca2, c2, o2, cb2, 
                           sg2, cprev_prox, nnext_prox, cprev_dist, nnext_dist)
        
        # calculate and set the disulfide dihedral angles
        self.chi1 = numpy.degrees(calc_dihedral(n1, ca1, cb1, sg1))
        self.chi2 = numpy.degrees(calc_dihedral(ca1, cb1, sg1, sg2))
        self.chi3 = numpy.degrees(calc_dihedral(cb1, sg1, sg2, cb2))
        self.chi4 = numpy.degrees(calc_dihedral(sg1, sg2, cb2, ca2))
        self.chi5 = numpy.degrees(calc_dihedral(sg2, cb2, ca2, n2))

        self.ca_distance = proteusPy.distance3d(self.ca_prox, self.ca_dist)
        self.torsion_array = numpy.array((self.chi1, self.chi2, self.chi3, 
                                        self.chi4, self.chi5))
        self.torsion_length = self.Torsion_length()

        # calculate and set the SS bond torsional energy
        self.compute_torsional_energy()

        # compute and set the local coordinates
        self.compute_local_coords()

    def set_chain_id(self, chain_id):
        self.chain_id = chain_id

    def set_positions(self, n_prox: Vector, ca_prox: Vector, c_prox: Vector,
                      o_prox: Vector, cb_prox: Vector, sg_prox: Vector, 
                      n_dist: Vector, ca_dist: Vector, c_dist: Vector,
                      o_dist: Vector, cb_dist: Vector, sg_dist: Vector,
                      c_prev_prox: Vector, n_next_prox: Vector,
                      c_prev_dist: Vector, n_next_dist: Vector
                      ):
        '''
        Set the atomic positions for all atoms in the Disulfide object.

        Parameters
        ----------
        n_prox : Vector
            Proximal N position
        ca_prox : Vector
            Proximal Ca position
        c_prox : Vector
            Proximal C' position
        o_prox : Vector
            Proximal O position
        cb_prox : Vector
            Proximal Cb position
        sg_prox : Vector
            Proximal Sg position
        n_dist : Vector
            Distal N position
        ca_dist : Vector
            Distal Ca position
        c_dist : Vector
            Distal C' position
        o_dist : Vector
            Distal O position
        cb_dist : Vector
            Distal Cb position
        sg_dist : Vector
            Distal Sg position
        c_prev_prox : Vector
            Proximal C'-1 position
        n_next_prox : Vector
            Proximal N+1 position
        c_prev_dist : Vector
            Distal C'-1 position
        n_next_dist : Vector
            Distal N+1 position
        '''
        
        # deep copy
        self.n_prox = n_prox.copy()
        self.ca_prox = ca_prox.copy()
        self.c_prox = c_prox.copy()
        self.o_prox = o_prox.copy()
        self.cb_prox = cb_prox.copy()
        self.sg_prox = sg_prox.copy()
        self.sg_dist = sg_dist.copy()
        self.cb_dist = cb_dist.copy()
        self.ca_dist = ca_dist.copy()
        self.n_dist = n_dist.copy()
        self.c_dist = c_dist.copy()
        self.o_dist = o_dist.copy()

        self.c_prev_prox = c_prev_prox.copy()
        self.n_next_prox = n_next_prox.copy()
        self.c_prev_dist = c_prev_dist.copy()
        self.n_next_dist = n_next_dist.copy()

    def set_dihedrals(self, chi1: float, chi2: float, chi3: float,
                    chi4: float, chi5: float):
        '''
        Set the disulfide's dihedral angles, Chi1-Chi5. -180 - 180 degrees.

        Parameters
        ----------
        chi1 : float
            Chi1
        chi2 : float
            Chi2
        chi3 : float
            Chi3
        chi4 : float
            Chi4
        chi5 : float
            Chi5
        '''
        
        self.chi1 = chi1
        self.chi2 = chi2
        self.chi3 = chi3
        self.chi4 = chi4
        self.chi5 = chi5
        #self.dihedrals = list([chi1, chi2, chi3, chi4, chi5])
        self.torsion_array = numpy.array(chi1, chi2, chi3, chi4, chi5)
        self.compute_torsional_energy()

    def set_name(self, namestr="Disulfide"):
        '''
        Set's the Disulfide's name

        Parameters
        ----------
        namestr : str, optional
            Name, by default "Disulfide"
        '''

        self.name = namestr

    def set_resnum(self, proximal: int, distal: int) -> None:
        '''
        Set the proximal and residue numbers for the Disulfide.

        Parameters
        ----------
        proximal : int
            Proximal residue number
        distal : int
            Distal residue number
        '''

        self.proximal = proximal
        self.distal = distal

    def Distance_RMS(self, other) -> float:
        '''
        Calculate the RMS distance between the internal coordinates
        of self and another Disulfide.

        Parameters
        ----------
        other : Disulfide
            Comparison Disulfide

        Returns
        -------
        float
            RMS distance (A).
        '''

        ic1 = self.internal_coords()
        ic2 = other.internal_coords()

        totsq = 0.0
        for i in range(12):
            p1 = ic1[i]
            p2 = ic2[i]
            totsq += math.dist(p1, p2)**2
        
        totsq /= 12

        return(math.sqrt(totsq))

    def compute_torsional_energy(self) -> float:
        '''
        Compute the approximate torsional energy for the Disulfide's
        conformation and sets its internal state.
        
        Returns
        -------
        float
            Energy (kcal/mol)
        
        '''
        # @TODO find citation for the ss bond energy calculation

        def torad(deg):
            return(numpy.radians(deg))

        chi1 = self.chi1
        chi2 = self.chi2
        chi3 = self.chi3
        chi4 = self.chi4
        chi5 = self.chi5

        energy = 2.0 * (cos(torad(3.0 * chi1)) + cos(torad(3.0 * chi5)))
        energy += cos(torad(3.0 * chi2)) + cos(torad(3.0 * chi4))
        energy += 3.5 * cos(torad(2.0 * chi3)) + 0.6 * cos(torad(3.0 * chi3)) + 10.1

        self.energy = energy
        return energy

    def compute_local_coords(self):
        """
        Compute the internal coordinates for a properly initialized Disulfide Object.
        
        Arguments: SS initialized Disulfide object
        
        Returns: None, modifies internal state of the input
        """

        turt = Turtle3D('tmp')
        # get the coordinates as numpy.array for Turtle3D use.
        cpp = self.c_prev_prox.get_array()
        nnp = self.n_next_prox.get_array()

        n = self.n_prox.get_array()
        ca = self.ca_prox.get_array()
        c = self.c_prox.get_array()
        cb = self.cb_prox.get_array()
        o = self.o_prox.get_array()
        sg = self.sg_prox.get_array()

        sg2 = self.sg_dist.get_array()
        cb2 = self.cb_dist.get_array()
        ca2 = self.ca_dist.get_array()
        c2 = self.c_dist.get_array()
        n2 = self.n_dist.get_array()
        o2 = self.o_dist.get_array()

        cpd = self.c_prev_dist.get_array()
        nnd = self.n_next_dist.get_array()
        
        turt.orient_from_backbone(n, ca, c, cb, ORIENT_SIDECHAIN)
        
        # internal (local) coordinates, stored as Vector objects
        # to_local returns numpy.array objects
        
        self._n_prox = Vector(turt.to_local(n))
        self._ca_prox = Vector(turt.to_local(ca))
        self._c_prox = Vector(turt.to_local(c))
        self._o_prox = Vector(turt.to_local(o))
        self._cb_prox = Vector(turt.to_local(cb))
        self._sg_prox = Vector(turt.to_local(sg))

        self._c_prev_prox = Vector(turt.to_local(cpp))
        self._n_next_prox = Vector(turt.to_local(nnp))
        self._c_prev_dist = Vector(turt.to_local(cpd))
        self._n_next_dist = Vector(turt.to_local(nnd))

        self._n_dist = Vector(turt.to_local(n2))
        self._ca_dist = Vector(turt.to_local(ca2))
        self._c_dist = Vector(turt.to_local(c2))
        self._o_dist = Vector(turt.to_local(o2))
        self._cb_dist = Vector(turt.to_local(cb2))
        self._sg_dist = Vector(turt.to_local(sg2))

    def build_model(self, turtle: Turtle3D):
        '''
        Build a model Disulfide based on the internal dihedral angles.
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinates. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.

        Parameters
        ----------
        turtle : Turtle3D
            turtle in orientation #1 (at Ca, headed toward Cb, with N on left)
        '''

        tmp = Turtle3D('tmp')
        tmp.copy_coords(turtle)

        n = Vector(0, 0, 0)
        ca = Vector(0, 0, 0)
        cb = Vector(0, 0, 0)
        c = Vector(0, 0, 0)

        self.ca_prox = tmp._position
        tmp.schain_to_bbone()
        n, ca, cb, c = build_residue(tmp)

        self.n_prox = n
        self.ca_prox = ca
        self.c_prox = c

        tmp.bbone_to_schain()
        tmp.move(1.53)
        tmp.roll(self.chi1)
        tmp.yaw(112.8)
        self.cb_prox = tmp._position

        tmp.move(1.86)
        tmp.roll(self.chi2)
        tmp.yaw(103.8)
        self.sg_prox = tmp._position

        tmp.move(2.044)
        tmp.roll(self.chi3)
        tmp.yaw(103.8)
        self.sg_dist = tmp._position

        tmp.move(1.86)
        tmp.roll(self.chi4)
        tmp.yaw(112.8)
        self.cb_dist = tmp._position

        tmp.move(1.53)
        tmp.roll(self.chi5)
        tmp.pitch(180.0)
        tmp.schain_to_bbone()
        n, ca, cb, c = build_residue(tmp)

        self.n_dist = n
        self.ca_dist = ca
        self.c_dist = c
        self.compute_torsional_energy()

    def Torsion_length(self) -> float:
        '''
        Compute the 5D Euclidean length of the Disulfide object
        and update the Disulfide internal state.

        Returns
        -------
        float
            Torsion length
        '''

        tors = self.torsion_array
        tors2 = tors * tors
        dist = math.sqrt(sum(tors2))

        self.torsion_length = dist
        return dist

    def Torsion_RMS(self, other) -> float:
        '''
        Calculate the 5D Euclidean distance between self and another Disulfide
        object. This is used to compare Disulfide Bond torsion angles to 
        determine their torsional similarity via a Euclidean distance metric.

        Parameters
        ----------
        other : Disulfide
            Comparison Disulfide

        Returns
        -------
        float
            RMS distance (degrees)

        Raises
        ------
        ProteusPyWarning
            Warning if other is wrong type.
        '''

        _p1 = self.torsion_array
        _p2 = other.torsion_array
        if (len(_p1) != 5 or len(_p2) != 5):
            raise ProteusPyWarning("--> distance5d() requires vectors of length 5!")
        d = math.dist(_p1, _p2)
        return d

    def Distance_RMS(self, other) -> float:
        '''
        Calculate the RMS distance between the internal coordinates between 
        two Disulfides

        Parameters
        ----------
        other : Disulfide
            Comparison Disulfide
        
        Returns
        -------
        float
            RMS distance (A)

        '''
        
        ic1 = self.internal_coords()
        ic2 = other.internal_coords()

        totsq = 0.0
        # only take coords for the proximal and distal disfulfides, not the 
        # prev/next residues.
        
        for i in range(12):
            p1 = ic1[i]
            p2 = ic2[i]
            totsq += math.dist(p1, p2)**2
        
        totsq /= 12

        return(math.sqrt(totsq))

# Class defination ends

def name_to_id(fname: str) -> str:
    '''
    Returns the PDB ID from the filename.

    Parameters
    ----------
    fname : str
        PDB filename

    Returns
    -------
    str
        PDB ID
    '''
    ent = fname[3:-4]
    return ent

def parse_ssbond_header_rec(ssbond_dict: dict) -> list:
    '''
    Parse the SSBOND dict returned by parse_pdb_header. 
    NB: Requires EGS-Modified BIO.parse_pdb_header.py.
    This is used internally.

    Parameters
    ----------
    ssbond_dict : dict
        the input SSBOND dict

    Returns
    -------
    list
        A list of tuples representing the proximal, 
        distal residue ids for the Disulfide.
    '''
    disulfide_list = []
    for ssb in ssbond_dict.items():
        disulfide_list.append(ssb[1])

    return disulfide_list

#
# function reads a comma separated list of PDB IDs and download the corresponding
# .ent files to the PDB_DIR global. 
# Used to download the list of proteins containing at least one SS bond
# with the ID list generated from: http://www.rcsb.org/
#

def Download_Disulfides(pdb_home=PDB_DIR, model_home=MODEL_DIR, 
                       verbose=False, reset=False) -> None:
    '''
    Reads a comma separated list of PDB IDs and downloads them
    to the pdb_home path. 

    Used to download the list of proteins containing at least one SS bond
    with the ID list generated from: http://www.rcsb.org/.

    This is the primary data loader for the proteusPy Disulfide 
    analysis package. The list of IDs represents files in the 
    RCSB containing > 1 disulfide bond, and it contains
    over 39000 structures. The total download takes about 12 hours. The
    function keeps track of downloaded files so it's possible to interrupt and
    restart the download without duplicating effort.

    Parameters
    ----------
    pdb_home : str, optional
        Path for downloaded files, by default PDB_DIR
    model_home : str, optional
        Path for extracted data, by default MODEL_DIR
    verbose : bool, optional
        Verbose, by default False
    reset : bool, optional
        Reset the downloaded file list, by default False

    Raises
    ------
    DisulfideIOException
        Fatal exception for file I/O
    '''
    start = time.time()
    donelines = []
    SS_done = []
    ssfile = None
    
    cwd = os.getcwd()
    os.chdir(pdb_home)

    pdblist = PDBList(pdb=pdb_home, verbose=verbose)
    ssfilename = f'{model_home}{SS_ID_FILE}'
    print(ssfilename)
    
    # list of IDs containing >1 SSBond record
    try:
        ssfile = open(ssfilename)
        Line = ssfile.readlines()
    except Exception:
        raise DisulfideIOException(f'Cannot open file: {ssfile}')

    for line in Line:
        entries = line.split(',')

    print(f'Found: {len(entries)} entries')
    completed = {'xxx'} # set to keep track of downloaded

    # file to track already downloaded entries.
    if reset==True:
        completed_file = open(f'{model_home}ss_completed.txt', 'w')
        donelines = []
        SS_DONE = []
    else:
        completed_file = open(f'{model_home}ss_completed.txt', 'w+')
        donelines = completed_file.readlines()

    if len(donelines) > 0:
        for dl in donelines[0]:
            # create a list of pdb id already downloaded
            SS_done = dl.split(',')

    count = len(SS_done) - 1
    completed.update(SS_done) # update the completed set with what's downloaded

    # Loop over all entries, 
    pbar = tqdm(entries, ncols=_PBAR_COLS)
    for entry in pbar:
        pbar.set_postfix({'Entry': entry})
        if entry not in completed:
            if pdblist.retrieve_pdb_file(entry, file_format='pdb', pdir=pdb_home):
                completed.update(entry)
                completed_file.write(f'{entry},')
                count += 1

    completed_file.close()

    end = time.time()
    elapsed = end - start

    print(f'Overall files processed: {count}')
    print(f'Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')
    os.chdir(cwd)
    return


def Extract_Disulfides(numb=-1, verbose=False, quiet=True, pdbdir=PDB_DIR, 
                        datadir=MODEL_DIR, picklefile=SS_PICKLE_FILE, 
                        torsionfile=SS_TORSIONS_FILE, 
                        problemfile=PROBLEM_ID_FILE,
                        dictfile=SS_DICT_PICKLE_FILE) -> None:
    '''
    This function creates .pkl files needed for the DisulfideLoader class. 
    The Disulfide objects are contained in a DisulfideList object and 
    Dict within these files. In addition, .csv files containing all of 
    the torsions for the disulfides and problem IDs are written.

    Parameters
    ----------
        numb:           number of entries to process, defaults to all
        verbose:        more messages
        quiet:          turns of DisulfideConstruction warnings
        pdbdir:         path to PDB files
        datadir:        path to resulting .pkl files
        picklefile:     name of the disulfide .pkl file
        torsionfile:    name of the disulfide torsion file .csv created
        problemfile:    name of the .csv file containing problem ids
        dictfile:       name of the .pkl file
    
    Examples:
    >>> from proteusPy.Disulfide import Disulfide 
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList
        
    Instantiate some variables. Note: the list is initialized with an iterable and a name (optional)

    >>> SS = Disulfide('tmp')
    >>> SSlist = DisulfideList([],'ss')
    >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)  # load the Disulfide database
    >>> SS = PDB_SS[0]
    >>> SS
    <Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A>

    >>> SS4yys = PDB_SS['4yys']     # returns a DisulfideList for ID 4yys
    >>> SS4yys
    [<Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A>, <Disulfide 4yys_56A_98A SourceID: 4yys Proximal: 56 A Distal: 98 A>, <Disulfide 4yys_156A_207A SourceID: 4yys Proximal: 156 A Distal: 207 A>, <Disulfide 4yys_22B_65B SourceID: 4yys Proximal: 22 B Distal: 65 B>, <Disulfide 4yys_56B_98B SourceID: 4yys Proximal: 56 B Distal: 98 B>, <Disulfide 4yys_156B_207B SourceID: 4yys Proximal: 156 B Distal: 207 B>]

    Make some empty disulfides
    >>> ss1 = Disulfide('ss1')
    >>> ss2 = Disulfide('ss2')

    Make a DisulfideList containing ss1, named 'tmp'
    >>> sslist = DisulfideList([ss1], 'tmp')
    >>> sslist.append(ss2)

    Extract the first disulfide and print it.
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
     Cα Distance: 4.502 Å
     Torsion length: 231.531 deg>

    Get a list of disulfides via slicing and display them:
    >>> subset = DisulfideList(PDB_SS[0:10],'subset')
    >>> subset.display_overlay()

    Take a screenshot. You can position the orientation, then close the window.
    >>> subset.screenshot(style='sb', fname='subset.png')  # save a screenshot.
    Saving file: subset.png
    Saved file: subset.png
    '''

    entrylist = []
    problem_ids = []
    bad = 0

    # we use the specialized list class DisulfideList to contain our disulfides
    # we'll use a dict to store DisulfideList objects, indexed by the structure ID
    All_ss_dict = {}
    All_ss_list = DisulfideList([],'PDB_SS')

    start = time.time()
    cwd = os.getcwd()

    # Build a list of PDB files in PDB_DIR that are readable. These files were downloaded
    # via the RCSB web query interface for structures containing >= 1 SS Bond.

    os.chdir(pdbdir)

    ss_filelist = glob.glob(f'*.ent')
    tot = len(ss_filelist)

    if verbose:
        print(f'PDB Directory {pdbdir} contains: {tot} files')

    # the filenames are in the form pdb{entry}.ent, I loop through them and extract
    # the PDB ID, with Disulfide.name_to_id(), then add to entrylist.

    for entry in ss_filelist:
        entrylist.append(name_to_id(entry))

    # create a dataframe with the following columns for the disulfide conformations 
    # extracted from the structure
    
    SS_df = pd.DataFrame(columns=Torsion_DF_Cols)

    # define a tqdm progressbar using the fully loaded entrylist list. 
    # If numb is passed then
    # only do the last numb entries.
    
    if numb > 0:
        pbar = tqdm(entrylist[:numb], ncols=_PBAR_COLS)
    else:
        pbar = tqdm(entrylist, ncols=_PBAR_COLS)

    # loop over ss_filelist, create disulfides and initialize them
    for entry in pbar:
        pbar.set_postfix({'ID': entry, 'Bad': bad}) # update the progress bar

        # returns an empty list if none are found.
        sslist = DisulfideList([], entry)
        sslist = load_disulfides_from_id(entry, model_numb=0, verbose=verbose, 
        								 quiet=quiet, pdb_dir=pdbdir)
        if len(sslist) > 0:
            for ss in sslist:
                All_ss_list.append(ss)
                new_row = [ss.pdb_id, ss.name, ss.proximal, ss.distal, 
                		  ss.chi1, ss.chi2, ss.chi3, ss.chi4, ss.chi5, 
                		  ss.energy, ss.ca_distance, ss.phiprox, 
                          ss.psiprox, ss.phidist, ss.psidist, ss.torsion_length]
                          
                # add the row to the end of the dataframe
                SS_df.loc[len(SS_df.index)] = new_row.copy() # deep copy
            All_ss_dict[entry] = sslist
        else:
            # at this point I really shouldn't have any bad non-parsible file
            bad += 1
            problem_ids.append(entry)
            os.remove(f'pdb{entry}.ent')
    
    if bad > 0:
        prob_cols = ['id']
        problem_df = pd.DataFrame(columns=prob_cols)
        problem_df['id'] = problem_ids

        print(f'Found and removed: {len(problem_ids)} problem structures.')
        print(f'Saving problem IDs to file: {datadir}{problemfile}')

        problem_df.to_csv(f'{datadir}{problemfile}')
    else:
        if verbose:
            print('No problems found.')
   
    # dump the all_ss array of disulfides to a .pkl file. ~520 MB.
    fname = f'{datadir}{picklefile}'
    print(f'Saving {len(All_ss_list)} Disulfides to file: {fname}')
    
    with open(fname, 'wb+') as f:
        pickle.dump(All_ss_list, f)

    # dump the all_ss array of disulfides to a .pkl file. ~520 MB.
    dict_len = len(All_ss_dict)
    fname = f'{datadir}{dictfile}'

    print(f'Saving {len(All_ss_dict)} Disulfide-containing PDB IDs to file: {fname}')

    with open(fname, 'wb+') as f:
        pickle.dump(All_ss_dict, f)

    # save the torsions
    fname = f'{datadir}{torsionfile}'
    print(f'Saving torsions to file: {fname}')

    SS_df.to_csv(fname)

    end = time.time()
    elapsed = end - start

    print(f'Disulfide Extraction complete! Elapsed time:\
    	 {datetime.timedelta(seconds=elapsed)} (h:m:s)')

    # return to original directory
    os.chdir(cwd)
    return

# NB - this only works with the EGS modified version of  BIO.parse_pdb_header.py
def load_disulfides_from_id(struct_name: str, 
                            pdb_dir = '.',
                            model_numb = 0, 
                            verbose = False,
                            quiet=False,
                            dbg = False) -> list:
    '''
    Loads the Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    *NB:* Requires EGS-Modified BIO.parse_pdb_header.py 

    Parameters
    ----------        
        struct_name: the name of the PDB entry.

        pdb_dir: path to the PDB files, defaults to PDB_DIR

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: a list of Disulfide objects initialized from the file.
    
    Example:
    PDB_DIR defaults to os.getenv('PDB').
    To load the Disulfides from the PDB ID 5rsa we'd use the following:
    
    >>> from proteusPy.Disulfide import Disulfide 
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList
    >>> PDB_DIR = '/Users/egs/PDB/good/'

    Instantiate a Disulfide list. Note: the list is initialized with an iterable and a name (optional)

    >>> SSlist = DisulfideList([],'ss')
    >>> SSlist = load_disulfides_from_id('5rsa', pdb_dir=PDB_DIR, verbose=False)
    >>> SSlist.pprint()
    <Disulfide 5rsa_26A_84A SourceID: 5rsa Proximal: 26 A Distal: 84 A 
    Conformation: (Χ1-Χ5):  -68.642°, -87.083°, -81.445°, -50.839° -66.097°  Energy: 1.758 kcal/mol 
    Cα Distance: 5.535 Å 
    Torsion length: 160.878 deg>
    <Disulfide 5rsa_40A_95A SourceID: 5rsa Proximal: 40 A Distal: 95 A 
    Conformation: (Χ1-Χ5):  -55.010°, -52.918°, -79.637°, -66.364° -61.091°  Energy: 0.711 kcal/mol 
    Cα Distance: 5.406 Å 
    Torsion length: 142.495 deg>
    <Disulfide 5rsa_58A_110A SourceID: 5rsa Proximal: 58 A Distal: 110 A 
    Conformation: (Χ1-Χ5):  -64.552°, -68.070°, -86.383°, -125.180° -46.499°  Energy: 3.102 kcal/mol 
    Cα Distance: 5.954 Å 
    Torsion length: 184.647 deg>
    <Disulfide 5rsa_65A_72A SourceID: 5rsa Proximal: 65 A Distal: 72 A 
    Conformation: (Χ1-Χ5):  -59.304°, -59.140°, 107.800°, 88.905° -81.312°  Energy: 3.802 kcal/mol 
    Cα Distance: 5.094 Å 
    Torsion length: 182.075 deg>
    
    '''

    i = 1
    proximal = distal = -1
    SSList = DisulfideList([], struct_name)
    _chaina = None
    _chainb = None

    parser = PDBParser(PERMISSIVE=True)
    
    # Biopython uses the Structure -> Model -> Chain hierarchy to organize
    # structures. All are iterable.

    structure = parser.get_structure(struct_name, file=f'{pdb_dir}pdb{struct_name}.ent')
    model = structure[model_numb]

    if verbose:
        print(f'-> load_disulfide_from_id() - Parsing structure: {struct_name}:')

    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code

    # list of tuples with (proximal distal chaina chainb)
    ssbonds = parse_ssbond_header_rec(ssbond_dict) 

    with warnings.catch_warnings():
        if quiet:
            #warnings.filterwarnings("ignore", category=DisulfideConstructionWarning)
            warnings.filterwarnings("ignore")
        for pair in ssbonds:
            # in the form (proximal, distal, chain)
            proximal = pair[0] 
            distal = pair[1]      
            chain1_id = pair[2]
            chain2_id = pair[3]

            if not proximal.isnumeric() or not distal.isnumeric():
                mess = f' -> Cannot parse SSBond record (non-numeric IDs):\
                 {struct_name} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}, ignoring.'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue
            else:
                proximal = int(proximal)
                distal = int(distal)
            
            if proximal == distal:
                mess = f' -> Cannot parse SSBond record (proximal == distal):\
                 {struct_name} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}, ignoring.'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue

            _chaina = model[chain1_id]
            _chainb = model[chain2_id]

            if (_chaina is None) or (_chainb is None):
                mess = f' -> NULL chain(s): {struct_name}: {proximal} {chain1_id}\
                 - {distal} {chain2_id}, ignoring!'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue

            if (chain1_id != chain2_id):
                if verbose:
                    mess = (f' -> Cross Chain SS for: Prox: {proximal} {chain1_id}\
                     Dist: {distal} {chain2_id}')
                    warnings.warn(mess, DisulfideConstructionWarning)
                    pass # was break

            try:
                prox_res = _chaina[proximal]
                dist_res = _chainb[distal]
                
            except KeyError:
                mess = f'Cannot parse SSBond record (KeyError): {struct_name} Prox:\
                  {proximal} {chain1_id} Dist: {distal} {chain2_id}, ignoring!'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue
            
            # make a new Disulfide object, name them based on proximal and distal
            # initialize SS bond from the proximal, distal coordinates
            
            if _chaina[proximal].is_disordered() or _chainb[distal].is_disordered():
                mess = f'Disordered chain(s): {struct_name}: {proximal} {chain1_id}\
                 - {distal} {chain2_id}, ignoring!'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue
            else:
                if verbose:
                    print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1_id}\
                     - {distal} {chain2_id}')
                ssbond_name = f'{struct_name}_{proximal}{chain1_id}_{distal}{chain2_id}'       
                new_ss = Disulfide(ssbond_name)
                new_ss.initialize_disulfide_from_chain(_chaina, _chainb, proximal,
                 distal, quiet=quiet)
                SSList.append(new_ss)
        i += 1
    return SSList

def check_header_from_file(filename: str, model_numb = 0, 
                            verbose = False, dbg = False) -> bool:

    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Parameters:
    ----------

        struct_name: the name of the PDB entry.

        pdb_dir: path to the PDB files, defaults to PDB_DIR

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: a list of Disulfide objects initialized from the file.
    Example:
      Assuming the PDB_DIR has the pdb5rsa.ent file in place calling:

      SS_list = []
      SS_list = load_disulfides_from_id('5rsa', verbose=True)

      loads the Disulfides from the file and initialize the disulfide objects, returning
      them in the result. '''

    i = 1
    proximal = distal = -1
    SSList = []
    _chaina = None
    _chainb = None

    parser = PDBParser(PERMISSIVE=True)
    
    # Biopython uses the Structure -> Model -> Chain hierarchy to organize
    # structures. All are iterable.

    structure = parser.get_structure('tmp', file=filename)
    struct_name = structure.get_id()
    
    model = structure[model_numb]

    if verbose:
        print(f'-> check_header_from_file() - Parsing file: {filename}:')

    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code

    # list of tuples with (proximal distal chaina chainb)
    ssbonds = parse_ssbond_header_rec(ssbond_dict) 

    for pair in ssbonds:
        # in the form (proximal, distal, chain)
        proximal = pair[0] 
        distal = pair[1]

        if not proximal.isnumeric() or not distal.isnumeric():
            if verbose:
                mess = f' ! Cannot parse SSBond record (non-numeric IDs):\
                 {struct_name} Prox:  {proximal} {chain1_id} Dist: {distal} {chain2_id}'
                warnings.warn(mess, DisulfideParseWarning)
            continue # was pass
        else:
            proximal = int(proximal)
            distal = int(distal)
        
        chain1_id = pair[2]
        chain2_id = pair[3]

        _chaina = model[chain1_id]
        _chainb = model[chain2_id]

        if (chain1_id != chain2_id):
            if verbose:
                mess = f' -> Cross Chain SS for: Prox: {proximal} {chain1_id} Dist:\
                       {distal} {chain2_id}'
                warnings.warn(mess, DisulfideParseWarning)
                pass # was break

        try:
            prox_res = _chaina[proximal]
            dist_res = _chainb[distal]
        except KeyError:
            print(f' ! Cannot parse SSBond record (KeyError): {struct_name} Prox:\
              <{proximal}> {chain1_id} Dist: <{distal}> {chain2_id}')
            continue
         
        # make a new Disulfide object, name them based on proximal and distal
        # initialize SS bond from the proximal, distal coordinates
        if (_chaina is not None) and (_chainb is not None):
            if _chaina[proximal].is_disordered() or _chainb[distal].is_disordered():
                continue
            else:
                if verbose:
                   print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1_id}\
                    - {distal} {chain2_id}')
        else:
            if dbg:
                print(f' -> NULL chain(s): {struct_name}: {proximal} {chain1_id}\
                 - {distal} {chain2_id}')
        i += 1
    return True

def check_header_from_id(struct_name: str, pdb_dir='.', model_numb=0, 
                            verbose=False, dbg=False) -> bool:
    '''
    Loads the Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Parameters
    ----------
    struct_name : str
        the name of the PDB entry.\n
    pdb_dir : str, optional
        path to the PDB files, defaults to PDB_DIR, by default '.'\n
    model_numb : int, optional
        model number to use, defaults to 0 for single structure files., by default 0 \n
    verbose : bool, optional
        Verbose, by default False\n
    dbg : bool, optional
        debugging flag, by default False

    Returns
    -------
    bool
        _description_
    '''

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(struct_name, 
    								 file=f'{pdb_dir}pdb{struct_name}.ent')
    model = structure[0]

    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code

    bondlist = []
    i = 0

    # get a list of tuples containing the proximal, distal residue IDs for 
    # all SSBonds in the chain.
    bondlist = parse_ssbond_header_rec(ssbond_dict)
    
    if len(bondlist) == 0:
        if (verbose):
            print(f'-> check_header_from_id(): no bonds found in bondlist.')
        return False
    
    for pair in bondlist:
        # in the form (proximal, distal, chain)
        proximal = pair[0]
        distal = pair[1]
        chain1 = pair[2]
        chain2 = pair[3]

        chaina = model[chain1]
        chainb = model[chain2]

        try:
            prox_residue = chaina[proximal]
            dist_residue = chainb[distal]
            
            prox_residue.disordered_select("CYS")
            dist_residue.disordered_select("CYS")

            if prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS':
                if (verbose):
                    print(f'build_disulfide() requires CYS at both residues:\
                     {prox_residue.get_resname()} {dist_residue.get_resname()}')
                return False
        except KeyError:
            if (dbg):
                print(f'Keyerror: {struct_name}: {proximal} {chain1} - {distal} {chain2}')
                return False
 
        if verbose:
            print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1} - {distal}\
             {chain2}')

        i += 1
    return True

def Check_chains(pdbid, pdbdir, verbose=True):
    '''Returns True if structure has multiple chains of identical length,\
     False otherwise'''

    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure(pdbid, file=f'{pdbdir}pdb{pdbid}.ent')
    
    # dictionary of tuples with SSBond prox and distal
    ssbond_dict = structure.header['ssbond']
    
    if verbose:
        print(f'ssbond dict: {ssbond_dict}')

    same = False
    model = structure[0]
    chainlist = model.get_list()

    if len(chainlist) > 1:
        chain_lens = []
        if verbose:
            print(f'multiple chains. {chainlist}')
        for chain in chainlist:
            chain_length = len(chain.get_list())
            chain_id = chain.get_id()
            if verbose:
                print(f'Chain: {chain_id}, length: {chain_length}')
            chain_lens.append(chain_length)

        if numpy.min(chain_lens) != numpy.max(chain_lens):
            same = False
            if verbose:
                print(f'chain lengths are unequal: {chain_lens}')
        else:
            same = True
            if verbose:
                print(f'Chains are equal length, assuming the same. {chain_lens}')
    return(same)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# End of file
