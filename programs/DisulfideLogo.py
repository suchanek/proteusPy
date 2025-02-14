# Disulfide Bond Analysis
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/2/23 -egs-
# Cα Cβ Sγ

import pandas as pd
import pyvista as pv
from pyvista import set_plot_theme

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.DisulfideLoader import Load_PDB_SS


def SS_DisplayTest(ss: Disulfide):
    ss.display(style="bs", single=False)
    ss.display(style="cpk")
    ss.display(style="sb", single=True)
    ss.display(style="pd", single=False)
    ss.screenshot(style="cpk", single=True, fname="cpk3.png", verbose=True)
    # ss.screenshot(style='sb', single=False, fname='sb3.png', verbose=True)
    return


if __name__ == "__main__":
    PDB_SS = None
    PDB_SS = Load_PDB_SS(verbose=True, subset=False)

    All_SS_list = PDB_SS.SSList
    ssMin, ssMax = All_SS_list.minmax_energy()
    minmaxlist = DisulfideList([ssMin, ssMax], "mm")
    # minmaxlist.display(style='bs', light=True)

    best = PDB_SS.get_by_name("2q7q_75D_140D")

    best.screenshot(single=True, style="cpk", shadows=False, fname="img/logo_cpk.png")
    best.screenshot(single=True, style="bs", shadows=False, fname="img/logo_bs.png")
    best.screenshot(single=True, style="sb", shadows=False, fname="img/logo_sb.png")

    exit()
