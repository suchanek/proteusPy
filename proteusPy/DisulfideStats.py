"""
This module provides statistical analysis functionality for disulfide bonds in the proteusPy package.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-12
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import PBAR_COLS, Torsion_DF_Cols, Distance_DF_Cols

_logger = create_logger(__name__)

class DisulfideStats:
    """Provides statistical analysis methods for Disulfide bonds, including torsion statistics,
    distance calculations, and data frame generation."""
    
    @staticmethod
    def build_distance_df(sslist, quiet=True) -> pd.DataFrame:
        """Create a dataframe containing the input DisulfideList Cα-Cα and Sg-Sg distances, energy.
        This can take several minutes for the entire database.

        :param sslist: List of Disulfide objects
        :param quiet: If True, suppress progress bar
        :return: DataFrame containing Ca distances
        """
        rows = []
        total_length = len(sslist)
        update_interval = max(1, total_length // 20)  # 5% of the list length

        if quiet:
            pbar = sslist
        else:
            pbar = tqdm(sslist, ncols=PBAR_COLS, leave=False)

        for ss in pbar:
            new_row = {
                "source": ss.pdb_id,
                "ss_id": ss.name,
                "proximal": ss.proximal,
                "distal": ss.distal,
                "energy": ss.energy,
                "ca_distance": ss.ca_distance,
                "cb_distance": ss.cb_distance,
                "sg_distance": ss.sg_distance,
            }
            rows.append(new_row)

        return pd.DataFrame(rows, columns=Distance_DF_Cols)

    @staticmethod
    def build_torsion_df(sslist, quiet=True) -> pd.DataFrame:
        """Create a dataframe containing the input DisulfideList torsional parameters,
        Cα-Cα and Sg-Sg distances, energy, and phi-psi angles.

        :param sslist: List of Disulfide objects
        :param quiet: If True, suppress progress bar
        :return: DataFrame containing the torsions
        """
        rows = []
        total_length = len(sslist)
        update_interval = max(1, total_length // 20)

        if quiet:
            pbar = sslist
        else:
            pbar = tqdm(sslist, ncols=PBAR_COLS, leave=False)

        for ss in pbar:
            new_row = {
                "source": ss.pdb_id,
                "ss_id": ss.name,
                "proximal": ss.proximal,
                "distal": ss.distal,
                "chi1": ss.chi1,
                "chi2": ss.chi2,
                "chi3": ss.chi3,
                "chi4": ss.chi4,
                "chi5": ss.chi5,
                "energy": ss.energy,
                "ca_distance": ss.ca_distance,
                "cb_distance": ss.cb_distance,
                "sg_distance": ss.sg_distance,
                "psi_prox": ss.psiprox,
                "phi_prox": ss.phiprox,
                "phi_dist": ss.phidist,
                "psi_dist": ss.psidist,
                "torsion_length": ss.torsion_length,
                "rho": ss.rho,
                "binary_class_string": ss.binary_class_string,
                "octant_class_string": ss.octant_class_string,
            }
            rows.append(new_row)

        return pd.DataFrame(rows, columns=Torsion_DF_Cols)

    @staticmethod
    def create_deviation_dataframe(sslist, verbose=False) -> pd.DataFrame:
        """Create a DataFrame with columns PDB_ID, SS_Name, Angle_Deviation, Distance_Deviation,
        Ca Distance from a list of disulfides.

        :param sslist: List of Disulfide objects
        :param verbose: Whether to display a progress bar
        :return: DataFrame containing the disulfide information
        """
        data = {
            "PDB_ID": [],
            "Resolution": [],
            "SS_Name": [],
            "Angle_Deviation": [],
            "Bondlength_Deviation": [],
            "Ca_Distance": [],
            "Sg_Distance": [],
        }

        if verbose:
            pbar = tqdm(sslist, desc="Processing...", leave=False)
        else:
            pbar = sslist

        for ss in pbar:
            data["PDB_ID"].append(ss.pdb_id)
            data["Resolution"].append(ss.resolution)
            data["SS_Name"].append(ss.name)
            data["Angle_Deviation"].append(ss.bond_angle_ideality)
            data["Bondlength_Deviation"].append(ss.bond_length_ideality)
            data["Ca_Distance"].append(ss.ca_distance)
            data["Sg_Distance"].append(ss.sg_distance)

        return pd.DataFrame(data)

    @staticmethod
    def calculate_torsion_statistics(sslist) -> tuple:
        """Calculate and return the torsion and distance statistics for the DisulfideList.

        :param sslist: List of Disulfide objects
        :return: A tuple containing two DataFrames:
                - tor_stats: DataFrame with mean and standard deviation for torsional parameters
                - dist_stats: DataFrame with mean and standard deviation for distance parameters
        """
        df = DisulfideStats.build_torsion_df(sslist)

        tor_cols = ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
        dist_cols = ["ca_distance", "cb_distance", "sg_distance", "energy", "rho"]
        tor_stats = {}
        dist_stats = {}

        def circular_mean(series):
            """Calculate the circular mean of a series of angles."""
            radians = np.deg2rad(series)
            sin_mean = np.sin(radians).mean()
            cos_mean = np.cos(radians).mean()
            return np.rad2deg(np.arctan2(sin_mean, cos_mean))

        for col in tor_cols[:5]:
            tor_stats[col] = {"mean": circular_mean(df[col]), "std": df[col].std()}

        tor_stats["torsion_length"] = {
            "mean": df["torsion_length"].mean(),
            "std": df["torsion_length"].std(),
        }

        for col in dist_cols:
            dist_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

        tor_stats = pd.DataFrame(tor_stats, columns=tor_cols)
        dist_stats = pd.DataFrame(dist_stats, columns=dist_cols)

        return tor_stats, dist_stats

    @staticmethod
    def extract_distances(sslist, distance_type="sg", comparison="less", cutoff=-1):
        """Extract and filter the distance values from the disulfide list based on the specified type and comparison.

        :param sslist: List of disulfide objects
        :param distance_type: Type of distance to extract ('sg' or 'ca')
        :param comparison: If 'less', return distances less than the cutoff value
        :param cutoff: Cutoff value for filtering distances
        :return: List of filtered distance values
        """
        distances = filtered_distances = []

        match distance_type:
            case "sg":
                distances = [ds.sg_distance for ds in sslist]
            case "ca":
                distances = [ds.ca_distance for ds in sslist]
            case _:
                raise ValueError("Invalid distance_type. Must be 'sg' or 'ca'.")

        if cutoff == -1.0:
            return distances

        if comparison == "greater":
            filtered_distances = [d for d in distances if d > cutoff]
        else:
            filtered_distances = [d for d in distances if d <= cutoff]

        return filtered_distances
