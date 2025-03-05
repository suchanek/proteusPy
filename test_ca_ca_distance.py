"""
Test script to demonstrate the Ca-Ca distance feature in disulfide_schematic.py
"""

import os
import matplotlib.pyplot as plt

import proteusPy as pp
from proteusPy.disulfide_schematic import (
    create_disulfide_schematic,
    create_disulfide_schematic_from_model,
)

# Load a disulfide from the database
PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)
ss = PDB_SS[0]  # Get the first disulfide

print(f"Selected disulfide: {ss.pdb_id} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}")
print(f"Ca-Ca distance: {ss.ca_distance:.2f} Å")

# Create a schematic with Ca-Ca distance line
fig, ax = create_disulfide_schematic(
    disulfide=ss,
    output_file="ca_ca_distance_test.png",
    show_angles=True,
    show_ca_ca_distance=True,
    style="publication"
)

print("Saved schematic to: ca_ca_distance_test.png")

# Create a model disulfide schematic with Ca-Ca distance line
fig2, ax2 = create_disulfide_schematic_from_model(
    chi1=-60,
    chi2=-60,
    chi3=-90,
    chi4=-60,
    chi5=-60,
    output_file="model_ca_ca_distance_test.png",
    show_angles=True,
    show_ca_ca_distance=True
)

print("Saved model schematic to: model_ca_ca_distance_test.png")

print("Test completed successfully!")
