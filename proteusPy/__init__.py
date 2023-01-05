#
# init for proteusPy package
# Copyright (c) 2022 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.
# Cα Cβ Sγ Χ1 - Χ5 Χ

__Version__ = "0.7dev"

import sys
import os
import glob
import warnings
import copy
import shutil

import pickle
import time
import datetime
import math
import numpy

import pandas as pd
from tqdm import tqdm
from numpy import cos
from collections import UserList

from Bio.PDB.vectors import calc_dihedral, calc_angle

from proteusPy.proteusGlobals import *
from proteusPy.proteusPyWarning import *

from proteusPy.DisulfideGlobals import *
from proteusPy.atoms import *
from proteusPy.DisulfideExceptions import DisulfideIOException, DisulfideConstructionWarning, DisulfideConstructionException

from proteusPy.turtle3D import Turtle3D
from proteusPy.turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN
from proteusPy.residue import build_residue, get_backbone_from_chain, to_alpha, to_carbonyl, to_nitrogen, to_oxygen

#from proteusPy.Disulfide import CysSelect, Disulfide
from proteusPy.Disulfide import name_to_id, todeg, torad, build_torsion_df, distance3d
from proteusPy.Disulfide import parse_ssbond_header_rec, DownloadDisulfides, ExtractDisulfides
from proteusPy.Disulfide import  cmap_vector, check_chains, Distance_RMS, Torsion_RMS

#from proteusPy.Disulfide import DisulfideList
from proteusPy.DisulfideLoader import DisulfideLoader

# end of file
