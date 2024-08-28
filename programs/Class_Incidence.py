"""
This script performs the following operations:

1. Sets up directory paths for saving plots and data using `pathlib.Path`.
2. Loads disulfide bond data from the PDB (Protein Data Bank) using the `Load_PDB_SS` function.
3. Describes the loaded disulfide bond data.
4. Generates and saves plots for binary to eight-class and binary to six-class incidence of disulfide bonds.
5. Prints a message indicating the start of plotting binary class incidence and saving the results.
6. Creates an empty DataFrame and generates a plot for count vs. class, saving the plot to the specified directory.

Directory Setup:
- `HOME`: The home directory of the user.
- `PDB`: The PDB directory, defaulting to `HOME/pdb` if the `PDB` environment variable is not set.
- `DATA_DIR`: Directory for data files within the PDB directory.
- `SAVE_DIR`: Directory for saving plots and documents.
- `OCTANT`, `SEXTANT`, `BINARY`: Subdirectories within `SAVE_DIR` for saving specific types of plots.

Functions:
- `Load_PDB_SS`: Loads the disulfide bond data from the PDB.
- `plot_binary_to_eightclass_incidence`: Generates and saves a plot for binary to eight-class incidence.
- `plot_binary_to_sixclass_incidence`: Generates and saves a plot for binary to six-class incidence.
- `plot_count_vs_class_df`: Generates and saves a plot for count vs. class using a DataFrame.

Usage:
- Ensure the required directories exist or will be created.
- Load the disulfide bond data.
- Generate and save the required plots.

Example:
    Run the script to generate and save the plots for disulfide bond incidence and count vs. class.

Dependencies:
- `os`
- `pathlib`
- `pandas`
- `proteusPy` (with modules: `Disulfide`, `DisulfideList`, `DisulfideLoader`, `Load_PDB_SS`)

Author: Eric G. Suchanek, PhD.
Last Modification: 8/27/2024
"""

import os
from pathlib import Path

import pandas as pd

from proteusPy import Disulfide, DisulfideList, DisulfideLoader, Load_PDB_SS

HOME = Path.home()
PDB = Path(os.getenv("PDB", HOME / "pdb"))

DATA_DIR = PDB / "data"
SAVE_DIR = HOME / "Documents" / "proteusPyDocs" / "classes"
REPO_DIR = HOME / "repos" / "proteusPy" / "data"

OCTANT = SAVE_DIR / "octant"
OCTANT.mkdir(parents=True, exist_ok=True)

SEXTANT = SAVE_DIR / "sextant"
SEXTANT.mkdir(parents=True, exist_ok=True)

BINARY = SAVE_DIR / "binary"
BINARY.mkdir(parents=True, exist_ok=True)


PDB_SS = Load_PDB_SS(subset=False, verbose=True)
PDB_SS.describe()

PDB_SS.plot_binary_to_eightclass_incidence(
    theme="light", save=True, verbose=True, savedir=OCTANT
)
PDB_SS.plot_binary_to_sixclass_incidence(
    theme="light", save=True, verbose=True, savedir=SEXTANT
)

print("Plotting binary class incidence and saving to", BINARY)

df = pd.DataFrame()
fig = PDB_SS.plot_count_vs_class_df(
    df,
    title="Binary",
    save=True,
    savedir=BINARY,
    verbose=True,
    base=2,
)

print("Program finished successfully")

# end of file
