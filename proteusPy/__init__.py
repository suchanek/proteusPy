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

from proteusPy.disulfide import DisulfideList, DisulfideLoader, CysSelect, Disulfide
from proteusPy.disulfide import name_to_id, todeg, torad, build_torsion_df, distance3d
from proteusPy.disulfide import parse_ssbond_header_rec, DownloadDisulfides, ExtractDisulfides, check_chains
from proteusPy.disulfide import render_disulfide
from proteusPy.disulfide import  cmap_vector

# end of file
