# Analysis of Disulfide Bonds in Proteins of Known Structure 
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/19/23 -egs-
# Cα Cβ Sγ

import math

import matplotlib
import matplotlib.pyplot as plt

import plotly_express as px
#import seaborn as sns

import proteusPy
from proteusPy import *
from proteusPy.Disulfide import *

from proteusPy.ProteusGlobals import *

import pandas as pd
import pyvista as pv
from pyvista import set_plot_theme

plt.style.use('dark_background')

# ipyvtklink
#pv.set_jupyter_backend('ipyvtklink')

set_plot_theme('document')

PDB_SS = DisulfideLoader(verbose=True, subset=False)

All_SS_list = PDB_SS.SSList

ssMin, ssMax = PDB_SS.SSList.minmax_energy()

#ssMin.screenshot(style='cpk', fname='ssmin_cpk.png')
#ssMin.screenshot(style='sb', fname='ssmin_sb.png')

ssMin.screenshot(style='bs', fname='ssmin_bs.png')
