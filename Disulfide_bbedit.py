#
# Implementation for a Disulfide Bond Class object.
# Based on the original C/C++ implementation by Eric G. Suchanek, PhD
# Part of the program Proteus, a program for the analysis and modeling of 
# protein structures with an emphasis on disulfide bonds
#

# Author: Eric G. Suchanek, PhD


import numpy
from numpy import cos

from Bio.PDB.vectors import Vector
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.PDBIO import Select
from Bio.PDB import PDBParser
from Bio.PDB import PDBList

import turtle3D
from turtle3D import Turtle3D
from Residue import build_residue

# global for initialization of dihedrals and energies
_FLOAT_INIT = -999.9

# global directory for PDB files
PDB_DIR = '/Users/egs/PDB/'

class CysSelect(Select):
    def accept_residue(self, residue):
        if residue.get_name() == 'CYS':
            return True
        else:
            return False

def torad(deg):
    return(numpy.radians(deg))

def todeg(rad):
    return(numpy.degrees(rad))

def parse_ssbond_header_rec(ssbond_dict: dict) -> list:
    '''
    Parse the SSBOND dict returned by parse_pdb_header. 
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        ssbond_dict: the input SSBOND dict
        chain_id: the string chain identifier
    Returns: a list of tuples representing the proximal, distal residue ids for the disulfide.

    '''

    disulfide_list = []
    prox = dist = -1

    for ssb in ssbond_dict.items():
        sbl = ssb[1]
        try:
            prox = int(sbl[0]) # will fail for weird residue codes
            dist = int(sbl[1])
            chna = sbl[2]
            chnb = sbl[3]
            if (prox == dist):
                pass

            disulfide_list.append((prox, dist, chna, chnb))
        except:
            print(f'Cannot parse SSBond record: Prox: {prox} {chna} Dist: {dist} {chnb}')
        return disulfide_list

# NB - this only works with the EGS modified version of  BIO.parse_pdb_header.py

def load_disulfides_from_id(struct_name: str, 
                            pdb_dir = PDB_DIR,
                            chain_id = 'A',
                            model_numb = 0, 
                            verbose = False) -> list:
    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
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


    parser = PDBParser(PERMISSIVE=True)

    # Biopython uses the Structure -> Model -> Chain hierarchy to organize
    # structures. All are iterable.

    structure = parser.get_structure(struct_name, file=f'{pdb_dir}pdb{struct_name}.ent')
    model = structure[model_numb]
    
    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code
    proximal = distal = -1

    DisulfideList = []
    i = 1

    if verbose:
        print(f'Parsing structure: {struct_name}:')

    for pair in ssbond_dict.items():
        # in the form (proximal, distal, chain)
        sbl = pair[1]
        
        try:
            proximal = int(sbl[0]) # will fail for weird residue codes
            distal = int(sbl[1])
            chain1 = sbl[2]
            chain2 = sbl[3]
            if (chain1 == chain_id):
                if (chain1 == chain2
                    chain = model[chain1]
                    # chainb = model[chain1]
            else:
                pass

        except:
            print(f'Cannot parse SSBond record: Prox: {proximal} {chain1} Dist: {distal} {chainb}')
            pass
 
        if verbose:
            print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1} - {distal} {chain2}')

        # make a new Disulfide object, name them based on proximal and distal
        ssbond_name = f'{struct_name}_{proximal}_{distal}'

        # initialize SS bond from the proximal, distal coordinates
        new_ss = Disulfide(ssbond_name)        
        new_ss.initialize_disulfide_from_chain(chain, chain_id, proximal, distal)

        DisulfideList.append(new_ss, chain_id)
        i += 1
    return DisulfideList

def check_header_from_file(filename: str,
                            model_numb = 0, 
                            verbose = False,
                            dbg=False) -> bool:
    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        struct_name: the name of the PDB entry.

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: True if the proximal and distal residues are CYS and there are no cross-chain SS bonds

    Example:
      Assuming the PDB_DIR has the pdb5rsa.ent file in place calling:

      SS_list = []
      goodfile = check_header_from_id('5rsa', verbose=True)

    '''
    pdblist = PDBList()

    parser = PDBParser(PERMISSIVE=True, QUIET=True)

    struct_name = 'tmp'
    
    structure = parser.get_structure(struct_name, file=filename)
    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code
    model = structure[0]
    bondlist = []

    # get a list of tuples containing the proximal, distal residue IDs for all SSBonds in the chain.
    bondlist = parse_ssbond_header_rec(ssbond_dict)
    
    if dbg:
        print(bondlist)
    
    if bondlist is None:
        return False
    
    # grab prox, distal from the bondlist, initialize a new Disulfide object
    # from the corresponding residues, add to the list

    i = 1

    if verbose:
        print(f'Checking structure: {filename}:')

    for pair in bondlist:
        # in the form (proximal, distal, chain)
        proximal = pair[0]
        distal = pair[1]
        chain1 = pair[2]
        chain2 = pair[3]
        
        # assert chain1 == chain2, f'{struct_name}: {proximal} {chain1} - {distal} {chain2}'

        chaina = model[chain1]
        chainb = model[chain2]
        """
        if (chain1 != chain2):
            chainb = model[chain2]  
        else:
            chainb = model[chain1]
        """


        try:
            prox_residue = chaina[proximal]
            dist_residue = chainb[distal]
            if prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS':
                if (verbose):
                    print(f'build_disulfide() requires CYS at both residues: {prox_residue.get_resname()} {dist_residue.get_resname()}')
                return False
        except KeyError:
            if (dbg):
                print(f'Keyerror: {struct_name}: {proximal} {chain1} - {distal} {chain2}')
                return False
 
        if verbose:
            print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1} - {distal} {chain2}')

    return True

def check_header_from_id(struct_name: str, 
                            pdb_dir = PDB_DIR,
                            model_numb = 0, 
                            verbose = False,
                            dbg = False) -> bool:
    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        struct_name: the name of the PDB entry.

        pdb_dir: path to the PDB files, defaults to PDB_DIR

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: True if the proximal and distal residues are CYS and there are no cross-chain SS bonds

    Example:
      Assuming the PDB_DIR has the pdb5rsa.ent file in place calling:

      SS_list = []
      goodfile = check_header_from_id('5rsa', verbose=True)

    '''
    pdblist = PDBList()

    parser = PDBParser(PERMISSIVE=True, QUIET=True)

    structure = parser.get_structure(struct_name, file=f'{pdb_dir}pdb{struct_name}.ent')
    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code
    model = structure[0]
    bondlist = []

    # get a list of tuples containing the proximal, distal residue IDs for all SSBonds in the chain.
    bondlist = parse_ssbond_header_rec(ssbond_dict)
    
    if len(bondlist) == 0:
        return False
    
    # grab prox, distal from the bondlist, initialize a new Disulfide object
    # from the corresponding residues, add to the list

    DisulfideList = []
    i = 1

    if verbose:
        print(f'Checking structure: {struct_name}:')

    for pair in bondlist:
        # in the form (proximal, distal, chain)
        proximal = pair[0]
        distal = pair[1]
        chain1 = pair[2]
        chain2 = pair[3]
        
        # assert chain1 == chain2, f'{struct_name}: {proximal} {chain1} - {distal} {chain2}'

        if (chain1 != chain2):
            if (verbose):
                print(f'Mixed chains: {struct_name}: {proximal} {chain1} - {distal} {chain2}')
            return False

        chain = model[chain1]

        try:
            prox_residue = chain[proximal]
            dist_residue = chain[distal]

        except KeyError:
            if (dbg):
                print(f'Keyerror: {struct_name}: {proximal} {chain1} - {distal} {chain2}')
                return False

        if prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS':
            if (verbose):
                print(f'build_disulfide() requires CYS at both residues: {prox_residue.get_resname()} {dist_residue.get_resname()}')
            return False

        if verbose:
            print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1} - {distal} {chain2}')

    return True

# α β γ Χ
# Cα 
# Cβ
# Sγ

# Class defination for a structure-based Disulfide Bond.

class Disulfide:
    """
    The Disulfide Bond is characterized by the atomic coordinates N, Cα, Cβ, C', Sγ 
    for both residues, the dihedral angles Χ1 - Χ5 for the disulfide bond conformation,
    a name, proximal resiude number and distal residue number, and conformational energy.
    All atomic coordinates are represented by the BIO.PDB.Vector class.
    """
    def __init__(self, name="SSBOND"):
        """
        Initialize the class. All positions are set to the origin. The optional string name may be passed.
        """
        # global coordinates for the Disulfide, typically as returned from the PDB file
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

        # local coordinates for the Disulfide, computed using the Turtle3D in 
        # Orientation #1 these are generally private.

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

        # Dihedral angles for the disulfide bond itself, set to _FLOAT_INIT
        self.chi1 = _FLOAT_INIT
        self.chi2 = _FLOAT_INIT
        self.chi3 = _FLOAT_INIT
        self.chi4 = _FLOAT_INIT
        self.chi5 = _FLOAT_INIT

        # I initialize an array for the torsions which will be used for comparisons
        self.dihedrals = numpy.array((_FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT), "d")
        self.name = name
        self.proximal = -1
        self.distal = -1
        self.energy = _FLOAT_INIT
        self.chain = 'A'

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
        s1 = f'<Disulfide: {self.name}\nProximal: {self.proximal} Distal: {self.distal}\n'
        return s1
    
    def repr_ss_coords(self):
        s2 = f'Proximal Coordinates:\n   N: {self.n_prox}\n   Cα: {self.ca_prox}\n   C: {self.c_prox}\n   O: {self.o_prox}\n   Cβ: {self.cb_prox}\n   Sγ: {self.sg_prox}\n\n'
        s3 = f'Distal Coordinates:\n   N: {self.n_dist}\n   Cα: {self.ca_dist}\n   C: {self.c_dist}\n   O: {self.o_dist}\n   Cβ: {self.cb_dist}\n   Sγ: {self.sg_dist}\n\n'
        stot = f'{s2}{s3}'
        return stot

    def repr_ss_conformation(self):
        s4 = f'Conformation:\n   Χ1: {self.chi1:.3f}°\n   Χ2: {self.chi2:.3f}°\n   Χ3: {self.chi3:.3f}°\n   Χ4: {self.chi4:.3f}°\n   Χ5: {self.chi5:.3f}°\n'
        s5 = f'Energy: {self.energy:.3f} kcal/mol>\n'
        stot = f'{s4}{s5}'
        return stot

    def repr_ss_local_coords(self):
        """
        Representation for the Disulfide class, internal coordinates.
        """
        s2i = f'Proximal Internal Coordinates:\n   N: {self._n_prox}\n   Cα: {self._ca_prox}\n   C: {self._c_prox}\n   O: {self._o_prox}\n   Cβ: {self._cb_prox}\n   Sγ: {self._sg_prox}\n'
        s3i = f'Distal Internal Coordinates:\n   N: {self._n_dist}\n   Cα: {self._ca_dist}\n   C: {self._c_dist}\n   O: {self._o_dist}\n   Cβ: {self._cb_dist}\n   Sγ: {self._sg_dist}\n'
        stot = f'{s2i}{s3i}'
        return stot

    def __repr__(self):
        """
        Representation for the Disulfide class
        """
        
        s1 = self.repr_ss_info()
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        res = f'{s1}{s2}{s3}{s4}'
        return res
   
    def reset(self):
        """
        Resets the internal state for a Disulfide. All positions are set 
        to the origin, torsions and energy reset.
        """

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

        # local coordinates for the Disulfide, computed using the Turtle3D in 
        # Orientation #1
        
        # these are generally private and are used when comparing/rendering

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

        self.chi1 = self.chi2 = self.chi3 = self.chi4 = self.chi5 = _FLOAT_INIT

        self.proximal = self.distal = -1
        self.energy = _FLOAT_INIT
        self.dihedrals = numpy.array((_FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT), "d")

    def initialize_disulfide_from_chain(self, chain, chain_id, proximal, distal):
        '''
        Initialize a new Disulfide object with atomic coordinates from the proximal and 
        distal coordinates, typically taken from a PDB file.

        Arguments: 
            chain: list of Residues in the model, eg: chain = model['A']
            proximal: proximal residue sequence ID
            distal: distal residue sequence ID
        
        Returns: none. The internal state is modified.
        '''

        # create a new Disulfide object

        chi1 = chi2 = chi3 = chi4 = chi5 = _FLOAT_INIT

        prox_residue = chain[proximal]
        dist_residue = chain[distal]

        if (prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS'):
            print(f'build_disulfide() requires CYS at both residues: {prox_residue.get_resname()} {dist_residue.get_resname()}')

        
        # set the objects proximal and distal values
        self.set_resnum(proximal, distal)
        self.set_chain_id(chain_id)

        # grab the coordinates for the proximal and distal residues as vectors so we can do math on them later

        # proximal residue
        n1 = prox_residue['N'].get_vector()
        ca1 = prox_residue['CA'].get_vector()
        c1 = prox_residue['C'].get_vector()
        o1 = prox_residue['O'].get_vector()
        cb1 = prox_residue['CB'].get_vector()
        sg1 = prox_residue['SG'].get_vector()

        # distal residue
        n2 = dist_residue['N'].get_vector()
        ca2 = dist_residue['CA'].get_vector()
        c2 = dist_residue['C'].get_vector()
        o2 = dist_residue['O'].get_vector()
        cb2 = dist_residue['CB'].get_vector()
        sg2 = dist_residue['SG'].get_vector()

        # update the positions and conformation
        self.set_positions(n1, ca1, c1, o1, cb1, sg1, n2, ca2, c2, o2, cb2, sg2)
        
        # calculate and set the disulfide dihedral angles
        self.chi1 = numpy.degrees(calc_dihedral(n1, ca1, cb1, sg1))
        self.chi2 = numpy.degrees(calc_dihedral(ca1, cb1, sg1, sg2))
        self.chi3 = numpy.degrees(calc_dihedral(cb1, sg1, sg2, cb2))
        self.chi4 = numpy.degrees(calc_dihedral(sg1, sg2, cb2, ca2))
        self.chi5 = numpy.degrees(calc_dihedral(sg2, cb2, ca2, n2))

        # calculate and set the SS bond torsional energy
        self.compute_disulfide_torsional_energy()

        # compute and set the local coordinates
        self.compute_local_disulfide_coords()

    def set_chain_id(self, chain_id):
        self.chain_id = chain_id

    def set_positions(self, n_prox: Vector, ca_prox: Vector, c_prox: Vector,
                      o_prox: Vector, cb_prox: Vector, sg_prox: Vector, 
                      n_dist: Vector, ca_dist: Vector, c_dist: Vector,
                      o_dist: Vector, cb_dist: Vector, sg_dist: Vector):
        '''
        Sets the atomic positions for all atoms in the disulfide bond.
        Arguments:
            n_prox
            ca_prox
            c_prox
            o_prox
            cb_prox
            sg_prox
            n_distal
            ca_distal
            c_distal
            o_distal
            cb_distal
            sg_distal
        Returns: None
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

    def set_conformation(self, chi1, chi2, chi3, chi4, chi5):
        '''
        Sets the 5 dihedral angles chi1-chi5 for the Disulfide object and computes the torsional energy.
        
        Arguments: chi, chi2, chi3, chi4, chi5 - Dihedral angles in degrees for the Disulfide conformation.
        Returns: None
        '''

        self.chi1 = chi1
        self.chi2 = chi2
        self.chi3 = chi3
        self.chi4 = chi4
        self.chi5 = chi5
        self.compute_disulfide_torsional_energy()

    def set_name(self, namestr="Disulfide"):
        '''
        Sets the Disulfide's name
        Arguments: (str)namestr
        Returns: none
        '''

        self.name = namestr

    def set_resnum(self, proximal, distal):
        '''
        Sets the Proximal and Distal Residue numbers for the Disulfide
        Arguments: 
            Proximal: Proximal residue number
            Distal: Distal residue number
        Returns: None
        '''

        self.proximal = proximal
        self.distal = distal

    def compute_disulfide_torsional_energy(self):
        '''
        Compute the approximate torsional energy for the Disulfide's conformation.
        Arguments: chi1, chi2, chi3, chi4, chi5 - the dihedral angles for the Disulfide
        Returns: Energy (kcal/mol)
        '''
        # @TODO find citation for the ss bond energy calculation
        chi1 = self.chi1
        chi2 = self.chi2
        chi3 = self.chi3
        chi4 = self.chi4
        chi5 = self.chi5

        energy = 2.0 * (cos(torad(3.0 * chi1)) + cos(torad(3.0 * chi5)))
        energy += cos(torad(3.0 * chi2)) + cos(torad(3.0 * chi4))
        energy += 3.5 * cos(torad(2.0 * chi3)) + 0.6 * cos(torad(3.0 * chi3)) + 10.1

        self.energy = energy

    def compute_local_disulfide_coords(self):
        """
        Compute the internal coordinates for a properly initialized Disulfide Object.
        Arguments: SS initialized Disulfide object
        Returns: None, modifies internal state of the input
        """

        turt = Turtle3D('tmp')
        # get the coordinates as numpy.array for Turtle3D use.
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

        turt.orient_from_backbone(n, ca, c, cb, turtle3D.ORIENT_SIDECHAIN)
        
        # internal (local) coordinates, stored as Vector objects
        # to_local returns numpy.array objects

        self._n_prox = Vector(turt.to_local(n))
        self._ca_prox = Vector(turt.to_local(ca))
        self._c_prox = Vector(turt.to_local(c))
        self._o_prox = Vector(turt.to_local(o))
        self._cb_prox = Vector(turt.to_local(cb))
        self._sg_prox = Vector(turt.to_local(sg))

        self._n_dist = Vector(turt.to_local(n2))
        self._ca_dist = Vector(turt.to_local(ca2))
        self._c_dist = Vector(turt.to_local(c2))
        self._o_dist = Vector(turt.to_local(o2))
        self._cb_dist = Vector(turt.to_local(cb2))
        self._sg_dist = Vector(turt.to_local(sg2))

    def build_disulfide_model(self, turtle: Turtle3D):
        """
        Build a model Disulfide based on the internal dihedral angles.
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinate state. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.

        Arguments: turtle: Turtle3D object properly oriented for the build.
        Returns: None. The Disulfide object's internal state is updated.
        """

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
        return

# Class defination ends

# End of file
    