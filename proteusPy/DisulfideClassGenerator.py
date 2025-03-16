#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate disulfide conformations for structural classes based on CSV data.

This script provides functionality to create disulfide conformations for different
structural classes using the mean and standard deviation values for each dihedral angle.
For each class, it generates all combinations of dihedral angles (mean-std, mean, mean+std)
for each of the 5 dihedral angles, resulting in 3^5 = 243 combinations per class.

Author: Eric G. Suchanek, PhD
Last Modification: 2025-03-15
"""

import itertools
import os
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.logger_config import create_logger

# Create a logger for this program
_logger = create_logger(__name__)
_logger.setLevel("WARNING")

# Constants
HOME_DIR: Path = Path.home()
PDB: Path = Path(os.getenv("PDB", HOME_DIR / "pdb"))
DATA_DIR: Path = PDB / "data"
SAVE_DIR: Path = HOME_DIR / "Documents" / "proteusPyDocs" / "classes"
MODULE_DIR: Path = HOME_DIR / "repos" / "proteusPy" / "proteusPy" / "data"
REPO_DIR: Path = HOME_DIR / "repos" / "proteusPy" / "data"
OCTANT: Path = SAVE_DIR / "octant"
BINARY: Path = SAVE_DIR / "binary"
MINIFORGE_DIR: Path = HOME_DIR / Path("miniforge3/envs")
MAMBAFORGE_DIR: Path = HOME_DIR / Path("mambaforge/envs")
VENV_DIR: Path = Path("lib/python3.12/site-packages/proteusPy/data")


class DisulfideClassGenerator:
    """
    A class for generating disulfide conformations for different structural classes
    based on mean and standard deviation values for dihedral angles. This creates
    243 combinations of dihedral angles for each class.
    """

    def __init__(self, csv_file: str = None, base: int = None):
        """
        Initialize the DisulfideClassGenerator.

        :param csv_file: Path to the CSV file containing class metrics.
            If provided, the CSV file will be loaded during initialization.
        :type csv_file: str, optional
        :param base: Optionally specify a base for disulfides.
        :type base: Optional[int]
        """
        self.df = None
        self.class_disulfides = {}
        self.base = base

        if csv_file:
            self.load_csv(csv_file)

    def load_csv(self, csv_file):
        """
        Load the CSV file containing class metrics.

        :param csv_file: Path to the CSV file.
        :type csv_file: str
        :return: Self for method chaining.
        :rtype: DisulfideClassGenerator
        """
        self.df = pd.read_csv(csv_file, dtype={"class": str, "class_str": str})
        return self

    def generate_for_class(self, class_id, use_class_str=True):
        """
        Generate disulfides for a specific structural class.

        :param class_id: The class ID or class string to generate disulfides for.
        :type class_id: str
        :param use_class_str: If True, match on class_str column, otherwise match on class column.
        :type use_class_str: bool
        :return: A list of Disulfide objects, or None if the class is not found.
        :rtype: DisulfideList or None
        :raises ValueError: If CSV file is not loaded.
        """
        if self.df is None:
            raise ValueError("CSV file not loaded. Call load_csv() first.")

        # Find the row for the specified class ID
        if use_class_str:
            row = self.df[self.df["class_str"] == class_id]
        else:
            row = self.df[self.df["class"] == class_id]

        if row.empty:
            print(f"Class ID {class_id} not found in the CSV file.")
            return None

        # Generate disulfides for the class
        _disulfide_list = self._generate_disulfides_for_class(row.iloc[0])

        # Update self.class_disulfides with the generated disulfides
        if _disulfide_list:
            self.class_disulfides[class_id] = _disulfide_list

        return _disulfide_list

    def generate_for_selected_classes(self, class_ids, use_class_str=True):
        """
        Generate disulfides for selected structural classes.

        :param class_ids: List of class IDs or class strings to generate disulfides for.
        :type class_ids: List[str]
        :param use_class_str: If True, match on class_str column, otherwise match on class column.
        :type use_class_str: bool
        :return: A dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If CSV file is not loaded.
        """
        if self.df is None:
            raise ValueError("CSV file not loaded. Call load_csv() first.")

        # Filter rows for selected class IDs
        if use_class_str:
            df_filtered = self.df[self.df["class_str"].isin(class_ids)]
            id_col = "class_str"
        else:
            df_filtered = self.df[self.df["class"].isin(class_ids)]
            id_col = "class"

        # Generate disulfides for each selected class
        class_disulfides = {}
        for _, row in df_filtered.iterrows():
            class_id = row[id_col]
            _disulfide_list = self._generate_disulfides_for_class(row)
            class_disulfides[class_id] = _disulfide_list

            # Update self.class_disulfides with the generated disulfides
            self.class_disulfides[class_id] = _disulfide_list

        return class_disulfides

    def generate_for_all_classes(self):
        """
        Generate disulfides for all structural classes in the CSV file.

        :return: A dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If CSV file is not loaded.
        """
        if self.df is None:
            raise ValueError("CSV file not loaded. Call load_csv() first.")

        # Generate disulfides for each class
        class_disulfides = {}
        for _, row in self.df.iterrows():
            class_id = row["class"]
            _disulfide_list = self._generate_disulfides_for_class(row)
            class_disulfides[class_id] = _disulfide_list

            # Update self.class_disulfides with the generated disulfides
            self.class_disulfides[class_id] = _disulfide_list

        return class_disulfides

    def _generate_disulfides_for_class(self, csv_row):
        """
        Generate disulfides for a structural class based on the mean and standard deviation
        values for each dihedral angle.

        :param csv_row: A row from the binary_class_metrics CSV file.
        :type csv_row: pd.Series
        :return: A list of Disulfide objects.
        :rtype: DisulfideList
        """
        class_id = csv_row["class"]
        class_str = csv_row["class_str"]

        # Extract mean and standard deviation values for each dihedral angle
        chi_means = [
            csv_row["chi1_mean"],
            csv_row["chi2_mean"],
            csv_row["chi3_mean"],
            csv_row["chi4_mean"],
            csv_row["chi5_mean"],
        ]

        chi_stds = [
            csv_row["chi1_std"],
            csv_row["chi2_std"],
            csv_row["chi3_std"],
            csv_row["chi4_std"],
            csv_row["chi5_std"],
        ]

        # Generate all combinations of dihedral angles
        disulfides = []
        base_str = ""

        # For each dihedral angle, consider mean-std, mean, and mean+std
        chi_values = []
        for i in range(5):
            mean = chi_means[i]
            std = chi_stds[i]
            chi_values.append([mean - std, mean, mean + std])

        # Generate all combinations (3^5 = 243 combinations)
        combinations = list(itertools.product(*chi_values))
        if self.base:
            base_str = "b" if self.base == 2 else "o"

        # Create a Disulfide object for each combination
        for i, combination in enumerate(combinations):
            name = f"{class_str}{base_str}_comb{i+1}"
            disulfide = Disulfide(name=name, torsions=list(combination))
            disulfides.append(disulfide)

        # Create a DisulfideList with all the generated disulfides
        _disulfide_list = DisulfideList(disulfides, f"Class_{class_id}_{class_str}")

        return _disulfide_list


def generate_disulfides_for_all_classes(
    csv_file: str, base=None
) -> Dict[str, DisulfideList]:
    """
    Generate disulfides for all structural classes in the CSV file.

    :param csv_file: Path to the CSV file containing class metrics.
    :type csv_file: str
    :param base: Optionally specify a base for disulfides.
    :type base: Optional[Any]
    :return: A dictionary mapping class IDs to DisulfideLists.
    :rtype: Dict[str, DisulfideList]
    """
    _generator = DisulfideClassGenerator(csv_file, base=base)
    return _generator.generate_for_all_classes()


def generate_disulfides_for_selected_classes(
    csv_file: str, class_ids: List[str], base: int = None, use_class_str: bool = False
) -> Dict[str, DisulfideList]:
    """
    Generate disulfides for selected structural classes in the CSV file.

    :param csv_file: Path to the CSV file containing class metrics.
    :type csv_file: str
    :param class_ids: List of class IDs to generate disulfides for.
    :type class_ids: List[str]
    :param base: Optionally specify a base for disulfides.
    :type base: Optional[Any]
    :return: A dictionary mapping class IDs to DisulfideLists.
    :rtype: Dict[str, DisulfideList]
    """
    _generator = DisulfideClassGenerator(csv_file, base=base)
    return _generator.generate_for_selected_classes(
        class_ids, use_class_str=use_class_str
    )


def generate_disulfides_for_class_from_csv(
    csv_file: str, class_id: str
) -> Union[DisulfideList, None]:
    """
    Generate disulfides for a specific structural class from the CSV file.

    :param csv_file: Path to the CSV file containing class metrics.
    :type csv_file: str
    :param class_id: The class ID or class string to generate disulfides for.
    :type class_id: str
    :return: A DisulfideList containing all combinations of
            dihedral angles for the given class, or None if
            the class ID is not found in the CSV file.
    :rtype: Union[DisulfideList, None]
    """
    _generator = DisulfideClassGenerator(csv_file)
    return _generator.generate_for_class(class_id, use_class_str=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate disulfides for structural classes."
    )
    parser.add_argument(
        "csv_file", help="Path to the CSV file containing class metrics."
    )

    parser.add_argument(
        "--class_id",
        help="Generate disulfides for a specific class id.",
    )

    parser.add_argument(
        "--base",
        help="Generate disulfides for a specific base.",
        default=None,
    )

    parser.add_argument(
        "--output", help="Output file to save the generated disulfides."
    )

    args = parser.parse_args()

    generator = DisulfideClassGenerator(args.csv_file)

    if args.class_id:
        disulfide_list = generator.generate_for_class(
            args.class_id, use_class_str=False
        )
        if disulfide_list:
            print(
                f"Generated {len(disulfide_list)} disulfides for class {args.class_id}."
            )

            if args.output:
                # Save the disulfide list to a file
                import pickle

                with open(args.output, "wb") as f:
                    pickle.dump(disulfide_list, f)
                print(f"Saved disulfides to {args.output}.")

    elif args.class_str:
        disulfide_list = generator.generate_for_class(
            args.class_str, use_class_str=True
        )
        if disulfide_list:
            print(
                f"Generated {len(disulfide_list)} disulfides for class {args.class_id}."
            )

            if args.output:
                # Save the disulfide list to a file
                import pickle

                with open(args.output, "wb") as f:
                    pickle.dump(disulfide_list, f)
                    print(f"Saved disulfides to {args.output}.")
    else:
        print("Please specify a class String using the --class_str argument.")

# end of file
