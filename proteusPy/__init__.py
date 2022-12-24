#
# init for proteusPy package
# Copyright (c) 2022 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.
# Cα Cβ Sγ Χ1 - Χ5 Χ

import sys
import os
import glob
import warnings
import copy

import numpy
import pickle
import time
import datetime

import pandas as pd
from tqdm import tqdm
from numpy import cos
from collections import UserList

__Version__ = "0.5dev"

from proteusPy.proteusGlobals import *
from proteusPy.DisulfideGlobals import *
from proteusPy.turtle3D import Turtle3D
from proteusPy.turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN

from proteusPy.disulfide import DisulfideList, DisulfideLoader, CysSelect, Disulfide

from proteusPy.disulfide import name_to_id, todeg, torad, build_torsion_df, distance3d, render_Disulfide, render_disulfide_panel
from proteusPy.disulfide import parse_ssbond_header_rec, DownloadDisulfides, ExtractDisulfides, check_chains
from proteusPy.DisulfideExceptions import DisulfideIOException, DisulfideConstructionWarning, DisulfideConstructionException
from proteusPy.residue import build_residue, get_backbone_from_chain, to_alpha, to_carbonyl, to_nitrogen, to_oxygen

from Bio.PDB import *
from Bio.PDB.vectors import calc_dihedral, calc_angle

# end of file
