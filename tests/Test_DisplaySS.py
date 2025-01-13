"""
This module, ``Test_DisplaySS.py``, is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds. Note:
depending on the speed of your hardware it can some time to load the full database and render
the disulfides.

Author: Eric G. Suchanek, PhD
Last revision: 10/9//2024
"""

# plyint: disable=C0116
# plyint: disable=C0103

import os
import shutil
import tempfile

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideList import DisulfideList
from proteusPy.DisulfideLoader import Load_PDB_SS

TMP = tempfile.mkdtemp()


def SS_DisplayTest(ss: Disulfide):
    ss.display(style="bs", single=True)
    ss.display(style="cpk", single=True)
    ss.display(style="sb", single=True)
    ss.display(style="pd", single=False)

    filename = os.path.join(TMP, "cpk3.png")
    ss.screenshot(style="cpk", single=True, fname=filename, verbose=True)

    filename = os.path.join(TMP, "sb3.png")
    ss.screenshot(style="sb", single=False, fname=filename, verbose=True)
    return


def SSlist_DisplayTest(sslist):
    sslist.display(style="cpk")
    sslist.display(style="bs")
    sslist.display(style="sb")
    sslist.display(style="pd")
    sslist.display(style="plain")


def main():
    """
    Program tests the proteusPy Disulfide rendering routines.
    Usage: run the program and close each window after manipulating it. The program
    will run through display styles and save a few files to /tmp. There should be no
    errors upon execution.
    """

    PDB_SS = None
    PDB_SS = Load_PDB_SS(verbose=True, subset=True)
    PDB_SS.describe()

    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]

    # SS_DisplayTest(ss)

    # get all disulfides for one structure. Make a
    # DisulfideList object to hold it

    ss4yss = DisulfideList([], "4yss")
    ss4yss = PDB_SS["4yys"]

    # SSlist_DisplayTest(ss4yss)

    # grab the last 12 disulfides
    sslist = DisulfideList([], "last12")

    sslist = PDB_SS[:12]
    SSlist_DisplayTest(sslist)

    print("--> Program complete!")
    shutil.rmtree(TMP)


if __name__ == "__main__":
    main()
