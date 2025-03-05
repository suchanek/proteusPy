"""
Test script for the modified disulfide_schematic module with actual residue names.

This script tests the changes made to display actual proximal and distal residue names
in the schematic diagrams.
"""

import os
import matplotlib.pyplot as plt

import proteusPy as pp
from proteusPy.disulfide_schematic import create_disulfide_schematic
from proteusPy.ProteusGlobals import DATA_DIR

def main():
    """Test the modified disulfide schematic with actual residue names."""
    # Create output directory if it doesn't exist
    output_dir = "schematic_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading disulfide from 5RSA...")
    # Load a disulfide from the database using a specific PDB ID
    pdb_ss = pp.load_disulfides_from_id("5RSA", pdb_dir=DATA_DIR)
    
    if len(pdb_ss) == 0:
        print("No disulfides found in 5RSA")
        return
    
    ss = pdb_ss[0]
    
    print(f"Selected disulfide: {ss.pdb_id} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}")
    
    # Create a schematic with the modified labels
    output_file = os.path.join(output_dir, "test_modified_schematic.png")
    fig, ax = create_disulfide_schematic(
        disulfide=ss,
        output_file=output_file,
        show_angles=True,
        show_ca_ca_distance=True,
        style="publication",
    )
    
    print(f"Saved schematic to: {output_file}")
    
    # Also create a model disulfide for comparison (should show default labels)
    output_file_model = os.path.join(output_dir, "test_model_schematic.png")
    fig_model, ax_model = create_disulfide_schematic(
        disulfide=None,  # This will create a model disulfide
        output_file=output_file_model,
        show_angles=True,
        show_ca_ca_distance=True,
        style="publication",
    )
    
    print(f"Saved model schematic to: {output_file_model}")

if __name__ == "__main__":
    main()
