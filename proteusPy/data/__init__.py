"""
Global declarations for the proteusPy package
"""

# init for proteusPy data module
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.

import os

_abspath = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR = f"{_abspath}/"

DATA_DIR = os.path.join(
    _abspath, ""
)  # os.path.join automatically adds the correct path separator
print(f"DATA_DIR: {DATA_DIR}")
