"""
This module provides statistical analysis functionality for disulfide bonds 
in the proteusPy package.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-19 23:17:44
"""

import logging
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from proteusPy.logger_config import create_logger, set_logger_level
from proteusPy.ProteusGlobals import PBAR_COLS, Distance_DF_Cols, Torsion_DF_Cols
from proteusPy.vector3D import calculate_bond_angle, rms_difference

# toggle_stream_handler("proteusPy.Vector3D", False)

_logger = create_logger(__name__)
set_logger_level("proteusPy.vector3D", "ERROR")


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
    def circular_mean(series):
        """Calculate the circular mean of a series of angles."""
        radians = np.deg2rad(series)
        sin_mean = np.sin(radians).mean()
        cos_mean = np.cos(radians).mean()
        return np.rad2deg(np.arctan2(sin_mean, cos_mean))

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

        def _circular_mean(series):
            """Calculate the circular mean of a series of angles."""
            radians = np.deg2rad(series)
            sin_mean = np.sin(radians).mean()
            cos_mean = np.cos(radians).mean()
            return np.rad2deg(np.arctan2(sin_mean, cos_mean))

        for col in tor_cols[:5]:
            tor_stats[col] = {"mean": _circular_mean(df[col]), "std": df[col].std()}

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

    @staticmethod
    def bond_angle_ideality(ss, verbose=False):
        """
        Calculate all bond angles for a disulfide bond and compare them to idealized angles.

        :param np.ndarray atom_coordinates: Array containing coordinates of atoms in the order:
            N1, CA1, C1, O1, CB1, SG1, N2, CA2, C2, O2, CB2, SG2
        :return: RMS difference between calculated bond angles and idealized bond angles.
        :rtype: float
        """

        atom_coordinates = ss.coords_array
        verbose = not ss.quiet
        if verbose:
            _logger.setLevel(logging.INFO)

        idealized_angles = {
            ("N1", "CA1", "C1"): 111.0,
            ("N1", "CA1", "CB1"): 108.5,
            ("CA1", "CB1", "SG1"): 112.8,
            ("CB1", "SG1", "SG2"): 103.8,  # This angle is for the disulfide bond itself
            ("SG1", "SG2", "CB2"): 103.8,  # This angle is for the disulfide bond itself
            ("SG2", "CB2", "CA2"): 112.8,
            ("CB2", "CA2", "N2"): 108.5,
            ("N2", "CA2", "C2"): 111.0,
        }

        # List of triplets for which we need to calculate bond angles
        # I am omitting the proximal and distal backbone angle N, Ca, C
        # to focus on the disulfide bond angles themselves.
        angle_triplets = [
            ("N1", "CA1", "C1"),
            ("N1", "CA1", "CB1"),
            ("CA1", "CB1", "SG1"),
            ("CB1", "SG1", "SG2"),
            ("SG1", "SG2", "CB2"),
            ("SG2", "CB2", "CA2"),
            ("CB2", "CA2", "N2"),
            ("N2", "CA2", "C2"),
        ]

        atom_indices = {
            "N1": 0,
            "CA1": 1,
            "C1": 2,
            "CB1": 4,
            "SG1": 5,
            "SG2": 11,
            "CB2": 10,
            "CA2": 7,
            "N2": 6,
            "C2": 8,
        }

        calculated_angles = []
        for triplet in angle_triplets:
            a = atom_coordinates[atom_indices[triplet[0]]]
            b = atom_coordinates[atom_indices[triplet[1]]]
            c = atom_coordinates[atom_indices[triplet[2]]]
            ideal = idealized_angles[triplet]
            try:
                angle = calculate_bond_angle(a, b, c)
            except ValueError as e:
                print(f"Error calculating angle for atoms {triplet}: {e}")
                return None
            calculated_angles.append(angle)
            if verbose:
                _logger.info(
                    "Calculated angle for atoms %s: %.2f, Ideal angle: %.2f",
                    triplet,
                    angle,
                    ideal,
                )

        # Convert idealized angles to a list
        idealized_angles_list = [
            idealized_angles[triplet] for triplet in angle_triplets
        ]

        # Calculate RMS difference
        rms_diff = rms_difference(
            np.array(calculated_angles), np.array(idealized_angles_list)
        )

        if verbose:
            _logger.info("RMS bond angle deviation: %.2f", rms_diff)

        return rms_diff

    @staticmethod
    def bond_length_ideality(ss, verbose=False):
        """
        Calculate bond lengths for a disulfide bond and compare them to idealized lengths.

        :param np.ndarray atom_coordinates: Array containing coordinates of atoms in the order:
            N1, CA1, C1, O1, CB1, SG1, N2, CA2, C2, O2, CB2, SG2
        :return: RMS difference between calculated bond lengths and idealized bond lengths.
        :rtype: float
        """

        atom_coordinates = ss.coords_array
        verbose = not ss.quiet
        if verbose:
            _logger.setLevel(logging.INFO)

        idealized_bonds = {
            ("N1", "CA1"): 1.46,
            ("CA1", "C1"): 1.52,
            ("CA1", "CB1"): 1.52,
            ("CB1", "SG1"): 1.86,
            ("SG1", "SG2"): 2.044,  # This angle is for the disulfide bond itself
            ("SG2", "CB2"): 1.86,
            ("CB2", "CA2"): 1.52,
            ("CA2", "C2"): 1.52,
            ("N2", "CA2"): 1.46,
        }

        # List of triplets for which we need to calculate bond angles
        # I am omitting the proximal and distal backbone angle N, Ca, C
        # to focus on the disulfide bond angles themselves.
        distance_pairs = [
            ("N1", "CA1"),
            ("CA1", "C1"),
            ("CA1", "CB1"),
            ("CB1", "SG1"),
            ("SG1", "SG2"),  # This angle is for the disulfide bond itself
            ("SG2", "CB2"),
            ("CB2", "CA2"),
            ("CA2", "C2"),
            ("N2", "CA2"),
        ]

        atom_indices = {
            "N1": 0,
            "CA1": 1,
            "C1": 2,
            "CB1": 4,
            "SG1": 5,
            "SG2": 11,
            "CB2": 10,
            "CA2": 7,
            "N2": 6,
            "C2": 8,
        }

        calculated_distances = []
        for pair in distance_pairs:
            a = atom_coordinates[atom_indices[pair[0]]]
            b = atom_coordinates[atom_indices[pair[1]]]
            ideal = idealized_bonds[pair]
            try:
                distance = math.dist(a, b)
            except ValueError as e:
                _logger.error("Error calculating bond length for atoms %s: %s", pair, e)
                return None
            calculated_distances.append(distance)
            if verbose:
                _logger.info(
                    "Calculated distance for atoms %s: %.2fA, Ideal distance: %.2fA",
                    pair,
                    distance,
                    ideal,
                )

        # Convert idealized distances to a list
        idealized_distance_list = [idealized_bonds[pair] for pair in distance_pairs]

        # Calculate RMS difference
        rms_diff = rms_difference(
            np.array(calculated_distances), np.array(idealized_distance_list)
        )

        if verbose:
            _logger.info(
                "RMS distance deviation from ideality for SS atoms: %.2f", rms_diff
            )

            # Reset logger level
            _logger.setLevel(logging.WARNING)

        return rms_diff

    # functions to calculate statistics and filter disulfide lists via pandas
    @staticmethod
    def calculate_std_cutoff(df, column, num_std=2):
        """
        Calculate cutoff based on standard deviation.

        :param df: DataFrame containing the deviations.
        :type df: pd.DataFrame
        :param column: Column name for which to calculate the cutoff.
        :type column: str
        :param num_std: Number of standard deviations to use for the cutoff.
        :type num_std: int
        :return: Cutoff value.
        :rtype: float
        """
        mean = df[column].mean()
        std = df[column].std()
        cutoff = mean + num_std * std
        return cutoff

    @staticmethod
    def calculate_percentile_cutoff(df, column, percentile=95):
        """
        Calculate cutoff based on percentile.

        :param df: DataFrame containing the deviations.
        :type df: pd.DataFrame
        :param column: Column name for which to calculate the cutoff.
        :type column: str
        :param percentile: Percentile to use for the cutoff.
        :type percentile: int
        :return: Cutoff value.
        :rtype: float
        """
        cutoff = np.percentile(df[column].dropna(), percentile)
        return cutoff

    @staticmethod
    def calculate_cutoff_from_percentile(
        sslist, percentile: float, verbose: bool = False
    ) -> dict:
        """
        Calculate the cutoff values for the standard deviation and percentile methods.

        :return: Dictionary containing the cutoff values for the standard deviation and percentile methods.
        :rtype: dict
        """
        from scipy.stats import norm

        # Set some parameters for the standard deviation and percentile methods
        std = 3.0
        dev_df = sslist.create_deviation_dataframe(verbose)

        # Calculate cutoffs using DisulfideStats methods
        distance_cutoff_std = DisulfideStats.calculate_std_cutoff(
            dev_df, "Bondlength_Deviation", num_std=std
        )
        angle_cutoff_std = DisulfideStats.calculate_std_cutoff(
            dev_df, "Angle_Deviation", num_std=std
        )
        ca_cutoff_std = DisulfideStats.calculate_std_cutoff(
            dev_df, "Ca_Distance", num_std=std
        )
        sg_cutoff_std = DisulfideStats.calculate_std_cutoff(
            dev_df, "Sg_Distance", num_std=std
        )

        # Percentile Method
        distance_cutoff_percentile = DisulfideStats.calculate_percentile_cutoff(
            dev_df, "Bondlength_Deviation", percentile=percentile
        )
        angle_cutoff_percentile = DisulfideStats.calculate_percentile_cutoff(
            dev_df, "Angle_Deviation", percentile=percentile
        )
        ca_cutoff_percentile = DisulfideStats.calculate_percentile_cutoff(
            dev_df, "Ca_Distance", percentile=percentile
        )
        sg_cutoff_percentile = DisulfideStats.calculate_percentile_cutoff(
            dev_df, "Sg_Distance", percentile=percentile
        )

        if verbose:
            print(
                f"Bond Length Deviation Cutoff ({std:.2f} Std Dev): {distance_cutoff_std:.2f}"
            )
            print(f"Angle Deviation Cutoff ({std:.2f} Std Dev): {angle_cutoff_std:.2f}")
            print(f"Ca Distance Cutoff ({std:.2f} Std Dev): {ca_cutoff_std:.2f}")
            print(f"Sg Distance Cutoff ({std:.2f} Std Dev): {sg_cutoff_std:.2f}")

            print(
                f"\nBond Length Deviation Cutoff ({percentile:.2f}th Percentile): {distance_cutoff_percentile:.2f}"
            )
            print(
                f"Angle Deviation Cutoff ({percentile:.2f}th Percentile): {angle_cutoff_percentile:.2f}"
            )
            print(
                f"Ca Distance Cutoff ({percentile:.2f}th Percentile): {ca_cutoff_percentile:.2f}"
            )
            print(
                f"Sg Distance Cutoff ({percentile:.2f}th Percentile): {sg_cutoff_percentile:.2f}"
            )

        # Calculate the Z-score for the percentile
        z_score = norm.ppf(percentile / 100.0)

        if verbose:
            print(
                f"The Z-score for the {percentile}th percentile is approximately {z_score:.3f}"
            )

        return {
            "distance_cutoff_std": distance_cutoff_std,
            "angle_cutoff_std": angle_cutoff_std,
            "ca_cutoff_std": ca_cutoff_std,
            "sg_cutoff_std": sg_cutoff_std,
            "distance_cutoff_percentile": distance_cutoff_percentile,
            "angle_cutoff_percentile": angle_cutoff_percentile,
            "ca_cutoff_percentile": ca_cutoff_percentile,
            "sg_cutoff_percentile": sg_cutoff_percentile,
        }


# EOF
