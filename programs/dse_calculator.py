import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from pyrosetta import *
from pyrosetta.teaching import *
from tqdm import tqdm


def download_pdb(pdbid):
    """Download PDB file if it doesn't exist"""
    pdbid = pdbid.lower()
    filename = f"{pdbid}.pdb"
    if not os.path.exists(filename):
        url = f"https://files.rcsb.org/download/{pdbid}.pdb"
        urllib.request.urlretrieve(url, filename)
    return filename


def get_disulfide_energy(pose, res1, res2):
    """Calculate disulfide torsional energy using PyRosetta's energy function"""
    # Create a copy of the pose to avoid modifying the original
    working_pose = Pose(pose)
    
    # Set up score function with disulfide term
    scorefxn = get_fa_scorefxn()
    scorefxn.set_weight(rosetta.core.scoring.dslf_fa13, 1.0)
    
    # Score the pose
    scorefxn(working_pose)
    
    # Get total energies
    score_terms = working_pose.energies().total_energies()
    torsional_energy = score_terms[rosetta.core.scoring.dslf_fa13]

    # Get chi angles for both residues
    chi_angles = []
    for res in [res1, res2]:
        angles = [working_pose.chi(i, res) for i in range(1, working_pose.residue(res).nchi() + 1)]
        chi_angles.append(angles)
    
    return torsional_energy, chi_angles[0], chi_angles[1]


def calculate_dse(pdbid, pose=None):
    """Calculate DSE for all disulfides in a structure"""
    if pose is None:
        # Download and load PDB if pose not provided
        pdb_file = download_pdb(pdbid)
        pose = pose_from_pdb(pdb_file)

    # Get chain information
    info = pose.pdb_info()
    if info is None:
        print("Error: No PDB info available")
        return None

    # Store results
    results = []
    processed = set()

    for i in range(1, pose.total_residue() + 1):
        # Get chain and residue info
        chain = info.chain(i)
        resnum = info.number(i)
        residue_i = pose.residue(i)

        # Skip if not CYS or already processed
        if residue_i.name3() != "CYS" or i in processed:
            continue

        # Look for disulfide partners
        for j in range(i + 1, pose.total_residue() + 1):
            residue_j = pose.residue(j)
            if residue_j.name3() == "CYS":
                # Calculate distance between sulfur atoms
                s_atom_i = residue_i.xyz("SG")
                s_atom_j = residue_j.xyz("SG")
                distance = s_atom_i.distance(s_atom_j)

                # Check if this is a disulfide bond
                if distance < 2.5:
                    j_chain = info.chain(j)
                    j_resnum = info.number(j)

                    # Calculate disulfide energy and get chi angles
                    total_strain, chi_1, chi_2 = get_disulfide_energy(pose, i, j)
                    
                    # Format chi angles for storage
                    chi_angles_1 = ','.join([f"{angle:.2f}" for angle in chi_1])
                    chi_angles_2 = ','.join([f"{angle:.2f}" for angle in chi_2])

                    # Store result with both proximal and distal information
                    results.append(
                        {
                            "pdbid": pdbid.upper(),
                            "proximal_residue": resnum,
                            "proximal_chain": chain,
                            "distal_residue": j_resnum,
                            "distal_chain": j_chain,
                            "dse_kcal_mol": total_strain,
                            "proximal_chi": chi_angles_1,
                            "distal_chi": chi_angles_2,
                        }
                    )

                    processed.add(i)
                    processed.add(j)

    # Create DataFrame
    df = pd.DataFrame(results)
    return df


def process_directory(directory):
    """Process all PDB files in a directory"""
    # Initialize PyRosetta once for all files
    pyrosetta.init(options="-mute core -ex1 -ex2 -use_input_sc")

    # Get list of PDB files
    pdb_files = [f for f in os.listdir(directory) if f.endswith(".ent")]

    # Process files with progress bar
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        try:
            # Get PDB ID from filename
            pdbid = os.path.splitext(pdb_file)[0]

            # Create pose directly from file
            pose = pose_from_pdb(os.path.join(directory, pdb_file))

            # Calculate DSE with the pose
            df = calculate_dse(pdbid, pose)

            if df is not None:
                # Save to CSV in same directory
                output_file = os.path.join(directory, f"{pdbid}_dse.csv")
                df.to_csv(output_file, index=False)
                # tqdm.write(f"Results saved to {output_file}")

        except Exception as e:
            tqdm.write(f"Error processing {pdb_file}: {str(e)}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python dse_calculator.py <directory>")
        sys.exit(1)

    directory = os.path.expandvars(sys.argv[1])  # Expand environment variables
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    process_directory(directory)


if __name__ == "__main__":
    main()
