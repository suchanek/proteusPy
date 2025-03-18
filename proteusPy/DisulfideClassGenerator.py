"""
Generate disulfide conformations for structural classes based on CSV data.

This module provides functionality to create disulfide conformations for different
structural classes using the mean and standard deviation values for each dihedral angle.
For each class, it generates all combinations of dihedral angles (mean-std, mean, mean+std)
for each of the 5 dihedral angles, resulting in 3^5 = 243 combinations per class.

Author: Eric G. Suchanek, PhD
Last Modification: 2025-03-17
"""

import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

# pylint: disable=C0103
import tqdm

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    BINARY_CLASS_METRICS_FILE,
    DATA_DIR,
    OCTANT_CLASS_METRICS_FILE,
)

# Create a logger for this program
_logger = create_logger(__name__)
_logger.setLevel("WARNING")


class DisulfideClassGenerator:
    """
    A class for generating disulfide conformations for different structural classes
    based on mean and standard deviation values for dihedral angles. This creates
    243 combinations of dihedral angles for each class.
    """

    def __init__(self, csv_file: str = None, base: int = None, verbose: bool = True):
        """
        Initialize the DisulfideClassGenerator.

        :param csv_file: Path to the CSV file containing class metrics.
        :type csv_file: str, optional
        :param base: Base for disulfides (2 for binary, 8 for octant).
        :type base: int, optional
        :param verbose: If True, log debug messages.
        :type verbose: bool
        :raises ValueError: If no valid data source is provided or schema is invalid.
        """
        self.df = None
        self.verbose = verbose
        self.class_disulfides: Dict[str, DisulfideList] = {}
        self.base = base
        binary_path = Path(DATA_DIR) / BINARY_CLASS_METRICS_FILE
        octant_path = Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE

        if csv_file:
            self.load_csv(csv_file)
        elif base == 2 and binary_path.exists():
            _logger.info("Loading binary metrics from %s", binary_path)
            self.df = pd.read_pickle(binary_path)
        elif base == 8 and octant_path.exists():
            _logger.info("Loading octant metrics from %s", octant_path)
            self.df = pd.read_pickle(octant_path)
        else:
            raise ValueError(
                "Provide csv_file or valid base (2 or 8) with existing file."
            )

        self._validate_df()
        self.df["class"] = self.df["class"].astype(str)
        self.df["class_str"] = self.df["class_str"].astype(str)

    def _validate_df(self):
        """Validate that the DataFrame has all required columns."""
        required = ["class", "class_str"] + [
            f"chi{i}_{stat}" for i in range(1, 6) for stat in ["mean", "std"]
        ]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        _logger.debug("DataFrame validated with columns: %s", list(self.df.columns))

    def load_csv(self, csv_file: str):
        """
        Load the CSV file containing class metrics.

        :param csv_file: Path to the CSV file.
        :type csv_file: str
        :return: Self for method chaining.
        :rtype: DisulfideClassGenerator
        """
        self.df = pd.read_csv(csv_file, dtype={"class": str, "class_str": str})
        return self

    def generate_for_class(
        self, class_id: str, use_class_str: bool = False
    ) -> Union[DisulfideList, None]:
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
            raise ValueError(
                "CSV file not loaded. Call load_csv() or initialize with valid base."
            )

        col = "class_str" if use_class_str else "class"
        row = self.df[self.df[col] == class_id]
        if row.empty:
            _logger.warning("Class ID %s not found in the data.", class_id)
            return None

        disulfide_list = self._generate_disulfides_for_class(row.iloc[0])
        self.class_disulfides[class_id] = disulfide_list
        _logger.info(
            "Generated %d disulfides for class %s", len(disulfide_list), class_id
        )
        return disulfide_list

    def generate_for_selected_classes(
        self, class_ids: List[str], use_class_str: bool = False
    ) -> Dict[str, DisulfideList]:
        """
        Generate disulfides for selected structural classes.
        Displays a tqdm progress bar if self.verbose is True.

        :param class_ids: List of class IDs or class strings to generate disulfides for.
        :type class_ids: List[str]
        :param use_class_str: If True, match on class_str column, otherwise match on class column.
        :type use_class_str: bool
        :return: A dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If CSV file is not loaded.
        """
        if self.df is None:
            raise ValueError("CSV file not loaded.")

        col = "class_str" if use_class_str else "class"
        df_filtered = self.df[self.df[col].isin(class_ids)]
        if df_filtered.empty:
            _logger.warning("No matching classes found for IDs: %s", class_ids)
            return {}

        class_disulfides = {}
        # Use tqdm for progress bar if verbose is True
        iterator = (
            tqdm.tqdm(
                df_filtered.iterrows(),
                total=len(df_filtered),
                desc="Generating selected disulfides",
            )
            if self.verbose
            else df_filtered.iterrows()
        )
        for _, row in iterator:
            class_id = row[col]
            disulfide_list = self._generate_disulfides_for_class(row)
            class_disulfides[class_id] = disulfide_list
            self.class_disulfides[class_id] = disulfide_list
        _logger.info("Generated disulfides for %d classes", len(class_disulfides))
        return class_disulfides

    def generate_for_all_classes(self) -> Dict[str, DisulfideList]:
        """
        Generate disulfides for all structural classes in the DataFrame.
        Displays a tqdm progress bar if self.verbose is True.

        :return: A dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If CSV file is not loaded.
        """
        if self.df is None:
            raise ValueError("CSV file not loaded.")

        class_disulfides = {}
        # Use tqdm for progress bar if verbose is True
        iterator = (
            tqdm.tqdm(
                self.df.iterrows(), total=len(self.df), desc="Generating disulfides"
            )
            if self.verbose
            else self.df.iterrows()
        )
        for _, row in iterator:
            class_id = row["class"]
            disulfide_list = self._generate_disulfides_for_class(row)
            class_disulfides[class_id] = disulfide_list
            self.class_disulfides[class_id] = disulfide_list
        _logger.info("Generated disulfides for all %d classes", len(class_disulfides))
        return class_disulfides

    def display(self, class_id: str, use_class_str: bool = False) -> None:
        """
        Display an overlay of all disulfides for a specific structural class.

        :param class_id: The class ID or class string to display disulfides for.
        :type class_id: str
        :param use_class_str: If True, match on class_str column, otherwise match on class column.
        :type use_class_str: bool
        :raises ValueError: If CSV file is not loaded or class is not found.
        """
        # First check if we already have this class generated
        if class_id in self.class_disulfides:
            disulfide_list = self.class_disulfides[class_id]
        else:
            # Generate if not already present
            disulfide_list = self.generate_for_class(class_id, use_class_str)
            if disulfide_list is None:
                raise ValueError(f"Class ID {class_id} not found in the data.")

        disulfide_list.display_overlay()

    def _generate_disulfides_for_class(self, csv_row) -> DisulfideList:
        """
        Generate disulfides for a structural class based on the mean and standard deviation
        values for each dihedral angle.

        :param csv_row: A row from the DataFrame.
        :type csv_row: pd.Series
        :return: A DisulfideList containing all 243 combinations.
        :rtype: DisulfideList
        """
        class_id = csv_row["class"]
        class_str = csv_row["class_str"]

        # Extract mean and standard deviation values for each dihedral angle
        chi_means = [csv_row[f"chi{i}_mean"] for i in range(1, 6)]
        chi_stds = [csv_row[f"chi{i}_std"] for i in range(1, 6)]

        # Generate all combinations (mean - std, mean, mean + std) for each angle
        chi_values = [
            [mean - std, mean, mean + std] for mean, std in zip(chi_means, chi_stds)
        ]
        combinations = list(itertools.product(*chi_values))  # 3^5 = 243 combinations

        # Determine base string
        base_str = "b" if self.base == 2 else "o" if self.base == 8 else ""

        # Create Disulfide objects for each combination
        disulfides = []
        for i, combo in enumerate(combinations):
            name = f"{class_str}{base_str}_comb{i+1}"
            disulfide = Disulfide(name=name, torsions=list(combo))
            disulfides.append(disulfide)

        # Return a DisulfideList
        return DisulfideList(disulfides, f"Class_{class_id}_{class_str}")


def generate_disulfides_for_all_classes(
    csv_file: str, base: int = None
) -> Dict[str, DisulfideList]:
    """
    Generate disulfides for all structural classes in the CSV file.

    :param csv_file: Path to the CSV file containing class metrics.
    :type csv_file: str
    :param base: Base for disulfides (2 or 8).
    :type base: int, optional
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
    :param base: Base for disulfides (2 or 8).
    :type base: int, optional
    :param use_class_str: If True, match on class_str column.
    :type use_class_str: bool
    :return: A dictionary mapping class IDs to DisulfideLists.
    :rtype: Dict[str, DisulfideList]
    """
    _generator = DisulfideClassGenerator(csv_file, base=base)
    return _generator.generate_for_selected_classes(
        class_ids, use_class_str=use_class_str
    )


def generate_disulfides_for_class_from_csv(
    csv_file: str = None, class_id: str = "", base: int = None
) -> Union[DisulfideList, None]:
    """
    Generate disulfides for a specific structural class from the CSV file.

    :param csv_file: Path to the CSV file containing class metrics.
    :type csv_file: str
    :param class_id: The class ID or class string to generate disulfides for.
    :type class_id: str
    :param base: Base for disulfides (2 or 8).
    :type base: int, optional
    :return: A DisulfideList or None if the class ID is not found.
    :rtype: Union[DisulfideList, None]
    """
    _generator = DisulfideClassGenerator(csv_file, base=base)
    if base == 2:
        return _generator.generate_for_class(class_id, use_class_str=True)
    elif base == 8:
        return _generator.generate_for_class(class_id, use_class_str=False)
    else:
        raise ValueError("Provide a valid base (2 or 8).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate disulfides for structural classes."
    )
    parser.add_argument("csv_file", help="Path to CSV file with class metrics.")
    parser.add_argument("--class_ids", nargs="+", help="List of class IDs to process.")
    parser.add_argument("--all", action="store_true", help="Process all classes.")
    parser.add_argument("--base", type=int, choices=[2, 8], help="Base (2 or 8).")
    parser.add_argument("--output", help="Output file for pickled results.")
    parser.add_argument("--verbose", action="store_true", help="Display progress bars.")
    args = parser.parse_args()

    generator = DisulfideClassGenerator(
        args.csv_file, base=args.base, verbose=args.verbose
    )
    if args.all:
        result = generator.generate_for_all_classes()
        print(f"Generated disulfides for {len(result)} classes.")
    elif args.class_ids:
        result = generator.generate_for_selected_classes(
            args.class_ids, use_class_str=False
        )
        print(f"Generated disulfides for {len(result)} classes.")
    else:
        print("Specify --class_ids or --all.")
        exit(1)

    if args.output and result:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved to {args.output}.")
