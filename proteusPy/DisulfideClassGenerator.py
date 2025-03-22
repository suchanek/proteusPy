"""
Generate disulfide conformations for structural classes based on CSV data.

This module provides functionality to create disulfide conformations for different
structural classes using the mean and standard deviation values for each dihedral angle.
For each class, it generates all combinations of dihedral angles (mean-std, mean, mean+std)
for each of the 5 dihedral angles, resulting in 3^5 = 243 combinations per class.

Author: Eric G. Suchanek, PhD
Last Modification: 2025-03-19 22:07:58
"""

# pylint: disable=C0301
# pylint: disable=C0103

import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
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

    @staticmethod
    def parse_class_string(class_str: str) -> tuple:
        """
        Parse a class string to determine its base and return the string without suffixes.

        :param class_str: The class string to parse (e.g., "11111", "11111b", "11111o", "+-+-+")
        :type class_str: str
        :return: A tuple containing (base, clean_string) where base is 2 for binary or 8 for octant
        :rtype: tuple
        :raises ValueError: If the class string format is invalid
        """
        if not isinstance(class_str, str):
            raise ValueError(f"Class string must be a string, got {type(class_str)}")

        # Initialize with default values
        base = 8  # Default to octant
        clean_string = class_str

        # Check for explicit suffix
        if len(class_str) == 6:
            if class_str[-1] == "b":
                base = 2
                clean_string = class_str[:5]
            elif class_str[-1] == "o":
                base = 8
                clean_string = class_str[:5]
            else:
                raise ValueError(f"Invalid class string suffix: {class_str}")
        # Check for binary class indicators (+ or -)
        elif "+" in class_str or "-" in class_str:
            base = 2
            clean_string = class_str[:5]

        return base, clean_string

    def __init__(
        self,
        csv_file: str = None,
        verbose: bool = True,
        precalc: bool = False,
        csv_base: int = None,
    ):
        """
        Initialize the DisulfideClassGenerator.

        :param csv_file: Path to the CSV file containing class metrics.
        :type csv_file: str, optional
        :param verbose: If True, log debug messages.
        :type verbose: bool
        :param precalc: If True, pre-calculate disulfides for all classes in the DataFrame.
        :type precalc: bool
        :param base: The base class to use, 2 or 8. Default is None.
        :type base: int, optional
        :raises ValueError: If no valid data source is provided or schema is invalid.
        """
        self.df = None
        self.binary_df = None
        self.octant_df = None
        self.verbose = verbose
        self.binary_class_disulfides: Dict[str, DisulfideList] = {}
        self.octant_class_disulfides: Dict[str, DisulfideList] = {}
        self.base = None  # Will be determined based on class_str

        binary_path = Path(DATA_DIR) / BINARY_CLASS_METRICS_FILE
        octant_path = Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE

        if csv_file:
            self.load_csv(csv_file, base=csv_base)
            self._validate_df()
        else:
            # Load both .pkl files at once if they exist
            if binary_path.exists():
                _logger.info("Loading binary metrics from %s", binary_path)
                self.binary_df = pd.read_pickle(binary_path)
                self.binary_df["class"] = self.binary_df["class"].astype(str)
                self.binary_df["class_str"] = self.binary_df["class_str"].astype(str)
                self.df = self.binary_df
                self._validate_df()

            if octant_path.exists():
                _logger.info("Loading octant metrics from %s", octant_path)
                self.octant_df = pd.read_pickle(octant_path)
                self.octant_df["class"] = self.octant_df["class"].astype(str)
                self.octant_df["class_str"] = self.octant_df["class_str"].astype(str)
                self.df = self.octant_df
                self._validate_df()

            # Default to octant if no specific file is provided
            self.df = self.octant_df if self.octant_df is not None else self.binary_df

        if self.df is None:
            raise ValueError(
                "No valid data source available. Provide csv_file or ensure .pkl files exist."
            )

        self.df["class"] = self.df["class"].astype(str)
        self.df["class_str"] = self.df["class_str"].astype(str)

        # Pre-calculate binary and octant dictionaries
        if precalc:
            if self.binary_df is not None:
                _logger.info("Pre-calculating binary class disulfides...")
                self._pre_calculate_class_disulfides(self.binary_df, is_binary=True)

            if self.octant_df is not None:
                _logger.info("Pre-calculating octant class disulfides...")
                self._pre_calculate_class_disulfides(self.octant_df, is_binary=False)

        return

    def _validate_df(self):
        """Validate that the DataFrame has all required columns."""
        required = ["class", "class_str"] + [
            f"chi{i}_{stat}" for i in range(1, 6) for stat in ["mean", "std"]
        ]

        # Only validate self.df if it's not the same as octant_df or binary_df
        if self.df is not None:
            missing = [col for col in required if col not in self.df.columns]
            if missing:
                raise ValueError(f"DataFrame missing required columns: {missing}")
            _logger.debug(
                "Custom DataFrame validated with columns: %s", list(self.df.columns)
            )

    def load_csv(self, csv_file: str, base: int = None) -> "DisulfideClassGenerator":
        """
        Load the CSV file containing class metrics.

        :param csv_file: Path to the CSV file.
        :type csv_file: str
        :param base: The base class to use, 2 or 8. Default is None.
        :type base: int, optional
        :return: Self for method chaining.
        :rtype: DisulfideClassGenerator
        """
        self.df = pd.read_csv(csv_file, dtype={"class": str, "class_str": str})
        self.df["class"] = self.df["class"].astype(str)
        self.df["class_str"] = self.df["class_str"].astype(str)
        self.base = base

        if base is not None:
            if base == 2:
                self.binary_df = self.df
            elif base == 8:
                self.octant_df = self.df
            else:
                raise ValueError(f"Invalid base class: {base}")

        return self

    def class_to_sslist(self, clsid: str, base: int = 8) -> np.ndarray:
        """
        Return the list of disulfides corresponding to the input `clsid`.
        This list is a list of disulfide identifiers, not the Disulfide objects themselves.

        :param clsid: The class name to extract. Must be a string
        in the format '11111' or '11111b' or '11111o'. The suffix 'b' or 'o' indicates
        binary or octant classes, respectively.
        :type clsid: str
        :param base: The base class to use, 2 or 8. Default is 8.
        :type base: int
        :param verbose: If True, display progress bars, by default False.
        :type verbose: bool
        :return: The list of disulfide bonds from the class. NB: this is the list
        of disulfide identifiers, not the Disulfide objects themselves.
        :rtype: DisulfideList
        :raises ValueError: If an invalid base value is provided.
        :raises KeyError: If the clsid is not found in the dictionary.
        """
        cls = clsid[:5]
        _base = base

        if not isinstance(clsid, str):
            _logger.error("Invalid class ID: %s", clsid)
            return

        match len(clsid):
            case 6:
                match clsid[-1]:
                    case "b":
                        _base = 2
                    case "o":
                        _base = 8
                    case _:
                        _logger.error("Invalid class ID suffix: %s", clsid)
                        return np.array([])

            case 5:
                match _base:
                    case 8:
                        _base = 8
                    case 2:
                        _base = 2
                    case _:
                        _logger.error("Invalid base: %d", base)
                        return np.array([])
            case _:
                _logger.error("Invalid class ID length: %s", clsid)
                return np.array([])

        try:
            if _base == 2:
                ss_ids = self.binary_class_disulfides[cls[:5]]
            else:  # _base == 8
                ss_ids = self.octant_class_disulfides[cls[:5]]
        except KeyError:
            _logger.error("Cannot find key %s in SSBond DB", clsid)
            return np.array([])

        return ss_ids

    def generate_for_class(self, class_id: str) -> Union[DisulfideList, None]:
        """
        Generate disulfides for a specific structural class.

        :param class_id: The class ID or class string to generate disulfides for.
        :type class_id: str
        :param use_class_str: If True, match on class_str column, otherwise match on class column.
        :rtype: DisulfideList or None
        :raises ValueError: If no valid data source is provided or class is not found.
        """

        # Check if self.df is None (for backward compatibility with tests)

        if self.df is None:
            raise ValueError(
                "No valid dataframe available for the specified class base."
            )

        # Determine the base from the class_id
        self.base, clean_string = self.parse_class_string(class_id)

        # Select the appropriate dataframe based on the base
        if self.base == 2:
            if self.binary_df is None:
                _logger.warning("Binary dataframe not available.")
                return None

            df_to_use = self.binary_df
            use_class_str = True

        else:  # self.base == 8
            if self.octant_df is None:
                _logger.warning("Octant dataframe not available.")
                return None

            df_to_use = self.octant_df
            use_class_str = False

        if df_to_use is None:
            raise ValueError(
                "No valid dataframe available for the specified class base."
            )

        col = "class_str" if use_class_str else "class"
        row = df_to_use[df_to_use[col] == class_id]

        # If not found, try with the clean string
        if row.empty and clean_string != class_id:
            row = df_to_use[df_to_use[col] == clean_string]

        if row.empty:
            _logger.warning("Class ID %s not found in the data.", class_id)
            return None

        disulfide_list = self._generate_disulfides_for_class(row.iloc[0])
        if self.base == 2:
            self.binary_class_disulfides[clean_string] = disulfide_list
        else:  # self.base == 8
            self.octant_class_disulfides[clean_string] = disulfide_list
        _logger.info(
            "Generated %d disulfides for class %s", len(disulfide_list), class_id
        )
        return disulfide_list

    def generate_for_selected_classes(
        self, class_ids: List[str]
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
        :raises ValueError: If no valid data source is provided or no classes are found.
        """
        if self.df is None:
            raise ValueError(
                "No valid dataframe available for the specified class base."
            )

        class_disulfides = {}
        # Use tqdm for progress bar if verbose is True
        iterator = (
            tqdm.tqdm(
                class_ids,
                total=len(class_ids),
                desc="Generating selected classes",
            )
            if self.verbose
            else class_ids
        )

        for class_id in iterator:
            disulfide_list = self.generate_for_class(class_id)
            if disulfide_list is not None:
                class_disulfides[class_id] = disulfide_list

        if not class_disulfides:
            _logger.warning("No matching classes found for IDs: %s", class_ids)
        else:
            _logger.info("Generated disulfides for %d classes", len(class_disulfides))

        return class_disulfides

    def generate_for_all_classes(self) -> Dict[str, DisulfideList]:
        """
        Generate disulfides for all structural classes in both binary and octant DataFrames.
        Displays a tqdm progress bar if self.verbose is True.

        :return: A dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If neither binary nor octant DataFrames are loaded.
        """
        if self.binary_df is None and self.octant_df is None:
            raise ValueError("Neither binary nor octant DataFrames are loaded.")

        class_disulfides = {}
        binary_count = 0
        octant_count = 0

        # Process binary classes if available
        if self.binary_df is not None:
            _logger.info("Generating disulfides for all binary classes...")
            self.base = 2  # Set base to binary

            # Use tqdm for progress bar if verbose is True
            iterator = (
                tqdm.tqdm(
                    self.binary_df.iterrows(),
                    total=len(self.binary_df),
                    desc="Generating binary consensus disulfides",
                )
                if self.verbose
                else self.binary_df.iterrows()
            )

            for _, row in iterator:
                class_id = row["class_str"]
                disulfide_list = self._generate_disulfides_for_class(row)
                class_disulfides[f"{class_id}b"] = (
                    disulfide_list  # Add 'b' suffix to distinguish
                )
                self.binary_class_disulfides[class_id] = disulfide_list
                binary_count += 1

            _logger.info("Generated disulfides for %d binary classes", binary_count)

        # Process octant classes if available
        if self.octant_df is not None:
            _logger.info("Generating disulfides for octant classes...")
            self.base = 8  # Set base to octant

            # Use tqdm for progress bar if verbose is True
            iterator = (
                tqdm.tqdm(
                    self.octant_df.iterrows(),
                    total=len(self.octant_df),
                    desc="Generating octant consensus disulfides",
                )
                if self.verbose
                else self.octant_df.iterrows()
            )

            for _, row in iterator:
                class_id = row["class"]
                disulfide_list = self._generate_disulfides_for_class(row)
                class_disulfides[f"{class_id}o"] = (
                    disulfide_list  # Add 'o' suffix to distinguish
                )
                self.octant_class_disulfides[class_id] = disulfide_list
                octant_count += 1

            _logger.info("Generated disulfides for %d octant classes", octant_count)

        total_count = binary_count + octant_count
        _logger.info(
            "Generated disulfides for a total of %d classes (%d binary, %d octant)",
            total_count,
            binary_count,
            octant_count,
        )

        return class_disulfides

    def describe(self, detailed: bool = False) -> None:
        """
        Pretty print the internal state of the DisulfideClassGenerator.

        This method displays information about the loaded DataFrames, generated disulfides,
        current base setting, and other internal state variables.

        :param detailed: If True, show more detailed information about DataFrames and disulfides.
        :type detailed: bool
        """
        print("\n" + "=" * 80)
        print(f"{'DisulfideClassGenerator State':^80}")
        print("=" * 80)

        # Basic information
        print(f"\nCurrent Base: {self.base if self.base is not None else 'Not set'}")
        print(f"Verbose Mode: {'Enabled' if self.verbose else 'Disabled'}")

        # DataFrame information
        print("\n" + "-" * 80)
        print(f"{'DataFrame Information':^80}")
        print("-" * 80)

        if self.binary_df is not None:
            binary_classes = len(self.binary_df)
            print(f"Binary Classes DataFrame: Loaded ({binary_classes} classes)")
            if detailed and binary_classes > 0:
                print("\nBinary DataFrame Sample:")
                print(self.binary_df.head(3).to_string())
        else:
            print("Binary Classes DataFrame: Not loaded")

        if self.octant_df is not None:
            octant_classes = len(self.octant_df)
            print(f"\nOctant Classes DataFrame: Loaded ({octant_classes} classes)")
            if detailed and octant_classes > 0:
                print("\nOctant DataFrame Sample:")
                print(self.octant_df.head(3).to_string())
        else:
            print("\nOctant Classes DataFrame: Not loaded")

        if self.df is not None:
            print(
                f"\nCurrent Working DataFrame: {'Binary' if self.df is self.binary_df else 'Octant' if self.df is self.octant_df else 'Custom'}"
            )
            print(f"Total Classes in Working DataFrame: {len(self.df)}")
        else:
            print("\nCurrent Working DataFrame: Not set")

        # Generated disulfides information
        print("\n" + "-" * 80)
        print(f"{'Generated Disulfides Information':^80}")
        print("-" * 80)

        binary_disulfides = len(self.binary_class_disulfides)
        print(f"Binary Class Disulfides: {binary_disulfides} classes generated")
        if detailed and binary_disulfides > 0:
            print("\nBinary Class IDs with generated disulfides:")
            for _, class_id in enumerate(list(self.binary_class_disulfides.keys())[:5]):
                disulfide_count = len(self.binary_class_disulfides[class_id])
                print(f"  {class_id}: {disulfide_count} disulfides")
            if binary_disulfides > 5:
                print(f"  ... and {binary_disulfides - 5} more classes")

        octant_disulfides = len(self.octant_class_disulfides)
        print(f"\nOctant Class Disulfides: {octant_disulfides} classes generated")
        if detailed and octant_disulfides > 0:
            print("\nOctant Class IDs with generated disulfides:")
            for _, class_id in enumerate(list(self.octant_class_disulfides.keys())[:5]):
                disulfide_count = len(self.octant_class_disulfides[class_id])
                print(f"  {class_id}: {disulfide_count} disulfides")
            if octant_disulfides > 5:
                print(f"  ... and {octant_disulfides - 5} more classes")

        # Memory usage estimation (rough)
        total_disulfides = sum(
            len(dl) for dl in self.binary_class_disulfides.values()
        ) + sum(len(dl) for dl in self.octant_class_disulfides.values())
        print(f"\nTotal Generated Disulfides: {total_disulfides}")

        print("\n" + "=" * 80 + "\n")

    def prepare_energy_data(self) -> pd.DataFrame:
        """
        Prepare a DataFrame containing energy values for each class.
        This data is suitable for creating box plots showing energy distribution by class.
        
        :return: DataFrame with columns for class_id and energy values
        :rtype: pd.DataFrame
        """
        energy_data = []
        
        # Process binary classes if available
        if self.binary_class_disulfides:
            for class_id, disulfide_list in self.binary_class_disulfides.items():
                for ss in disulfide_list:
                    energy_data.append({
                        "class": class_id,
                        "class_str": class_id,
                        "energy": ss.energy,
                        "base": 2
                    })
        
        # Process octant classes if available
        if self.octant_class_disulfides:
            for class_id, disulfide_list in self.octant_class_disulfides.items():
                for ss in disulfide_list:
                    energy_data.append({
                        "class": class_id,
                        "class_str": class_id,
                        "energy": ss.energy,
                        "base": 8
                    })
        
        # Create DataFrame from collected data
        energy_df = pd.DataFrame(energy_data)
        
        if not energy_df.empty:
            _logger.info("Created energy DataFrame with %d entries", len(energy_df))
        else:
            _logger.warning("No energy data available. Generate disulfides first.")
            
        return energy_df
    
    def plot_energy_by_class(
        self,
        base: int = None,
        title: str = "Energy Distribution by Class",
        theme: str = "auto",
        save: bool = False,
        savedir: str = ".",
        verbose: bool = False
    ) -> None:
        """
        Create a box plot showing energy distribution by class_id.
        
        :param base: The base class to use (2 for binary, 8 for octant). If None, use all available data.
        :type base: int, optional
        :param title: Title for the plot
        :type title: str
        :param theme: Theme to use for the plot ('auto', 'light', or 'dark')
        :type theme: str
        :param save: Whether to save the plot
        :type save: bool
        :param savedir: Directory to save the plot to
        :type savedir: str
        :param verbose: Whether to display verbose output
        :type verbose: bool
        """
        from proteusPy.DisulfideVisualization import DisulfideVisualization
        
        # Prepare energy data
        energy_df = self.prepare_energy_data()
        
        if energy_df.empty:
            _logger.warning("No energy data available. Generate disulfides first.")
            return
        
        # Filter by base if specified
        if base is not None:
            if base not in [2, 8]:
                raise ValueError("Base must be 2 (binary) or 8 (octant)")
            
            energy_df = energy_df[energy_df["base"] == base]
            base_str = "Binary" if base == 2 else "Octant"
            plot_title = f"{base_str} Class {title}"
        else:
            plot_title = title
        
        # Create the box plot
        DisulfideVisualization.plot_energy_by_class(
            energy_df,
            title=plot_title,
            theme=theme,
            save=save,
            savedir=savedir,
            verbose=verbose
        )
    
    def display(
        self,
        class_id: str,
        screenshot: bool = False,
        movie: bool = False,
        fname: str = "ss_overlay.png",
        theme: str = "auto",
        winsize: tuple = (1024, 1024),
        verbose: bool = False,
    ) -> None:
        """
        Display an overlay of all disulfides for a specific structural class.

        :param class_id: The class ID or class string to display disulfides for.
        :type class_id: str
        :param screenshot: If True, save a screenshot of the overlay.
        :type screenshot: bool
        :param movie: If True, save a movie of the overlay.
        :type movie: bool
        :param fname: Filename for screenshot or movie.
        :type fname: str
        :param theme: Color theme for the overlay.
        :type theme: str
        :param winsize: Window size for the overlay.
        :type winsize: tuple
        :param verbose: If True, display verbose output.
        :type verbose: bool
        :raises ValueError: If no valid data source is provided or class is not found.
        """
        # First check if we already have this class generated
        disulfide_list = None
        _base, class_str = self.parse_class_string(class_id)

        # Check if the class is already in the dictionaries
        if _base == 2 and class_str in self.binary_class_disulfides:
            disulfide_list = self.binary_class_disulfides[class_str]
        elif _base == 8 and class_str in self.octant_class_disulfides:
            disulfide_list = self.octant_class_disulfides[class_str]
        else:
            # Try to generate if not already present
            try:
                if _base == 2:
                    disulfide_list = self.generate_for_class(class_str)
                else:  # _base == 8
                    disulfide_list = self.generate_for_class(class_str)

                # If generation failed, try with the original class_id
                if disulfide_list is None:
                    disulfide_list = self.generate_for_class(class_id)

                # If still None, try to find a similar class
                if disulfide_list is None:
                    raise ValueError(
                        f"Class ID {class_id} not found in the data and no similar class found."
                    )

            except Exception as e:
                raise ValueError(
                    f"Failed to generate disulfides for class {class_id}: {str(e)}"
                ) from e

        # Now we have a valid disulfide_list, display it
        disulfide_list.display_overlay(
            screenshot=screenshot,
            movie=movie,
            verbose=verbose,
            fname=fname,
            winsize=winsize,
            light=theme,
        )

    def _pre_calculate_class_disulfides(
        self, df: pd.DataFrame, is_binary: bool
    ) -> None:
        """
        Pre-calculate disulfides for all classes in the given dataframe.

        :param df: The dataframe containing class metrics.
        :type df: pd.DataFrame
        :param is_binary: Whether the dataframe contains binary (True) or octant (False) classes.
        :type is_binary: bool
        """
        # Set the base for generating disulfides
        self.base = 2 if is_binary else 8

        # Use tqdm for progress bar if verbose is True
        iterator = (
            tqdm.tqdm(
                df.iterrows(),
                total=len(df),
                desc=f"Pre-calculating {'binary' if is_binary else 'octant'} disulfides",
            )
            if self.verbose
            else df.iterrows()
        )

        # Generate disulfides for each class
        for _, row in iterator:
            class_id = row["class"]
            disulfide_list = self._generate_disulfides_for_class(row)

            # Store in the appropriate dictionary
            if is_binary:
                self.binary_class_disulfides[class_id] = disulfide_list
            else:
                self.octant_class_disulfides[class_id] = disulfide_list

        _logger.info(
            "Pre-calculated %d %s disulfides",
            len(
                self.binary_class_disulfides
                if is_binary
                else self.octant_class_disulfides
            ),
            "binary" if is_binary else "octant",
        )

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
        # base_str = "" if self.base == 2 else "o" if self.base == 8 else ""

        base_str = ""
        # Create Disulfide objects for each combination
        disulfides = []
        for i, combo in enumerate(combinations):
            name = f"{class_str}{base_str}_comb{i+1}"
            disulfide = Disulfide(name=name, torsions=list(combo))
            disulfides.append(disulfide)

        # Return a DisulfideList
        return DisulfideList(disulfides, f"Class_{class_id}_{class_str}")

    @staticmethod
    def display_class_disulfides(
        class_string,
        light="auto",
        screenshot=False,
        movie=False,
        verbose=False,
        fname="ss_overlay.png",
        winsize=(1024, 1024),
    ):
        """
        Display disulfides belonging to a specific class using DisulfideClassGenerator.

        :param class_string: The binary or octant class string (e.g., "00000" for binary,
        "22632" for octant)
        :type class_string: str
        :param light: The background color theme ("auto", "light", or "dark")
        :type light: str
        :param screenshot: Whether to save a screenshot
        :type screenshot: bool
        :param movie: Whether to save a movie
        :type movie: bool
        :param verbose: Whether to display verbose output
        :type verbose: bool
        :param fname: Filename to save for the movie or screenshot
        :type fname: str
        :param winsize: Window size for the display (width, height)
        :type winsize: tuple
        """
        if verbose:
            print("Creating DisulfideClassGenerator...")

        _generator = None

        _base, _class_string = DisulfideClassGenerator.parse_class_string(class_string)

        # Create a DisulfideClassGenerator
        _generator = DisulfideClassGenerator(verbose=verbose)

        if verbose:
            print(f"Displaying disulfides for class {class_string}...")

        # Display the disulfides for the specified class
        try:
            _generator.display(
                _class_string,
                screenshot=screenshot,
                movie=movie,
                fname=fname,
                theme=light,
                winsize=winsize,
                verbose=verbose,
            )
        except ValueError as e:
            print(f"Error: {e}")


# class ends


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate disulfides for structural classes."
    )
    parser.add_argument("csv_file", help="Path to CSV file with class metrics.")
    parser.add_argument("--class_ids", nargs="+", help="List of class IDs to process.")
    parser.add_argument("--all", action="store_true", help="Process all classes.")
    parser.add_argument("--output", help="Output file for pickled results.")
    parser.add_argument("--verbose", action="store_true", help="Display progress bars.")
    args = parser.parse_args()

    generator = DisulfideClassGenerator(args.csv_file, verbose=args.verbose)
    if args.all:
        result = generator.generate_for_all_classes()
        print(f"Generated disulfides for {len(result)} classes.")
    elif args.class_ids:
        result = generator.generate_for_selected_classes(args.class_ids)
        print(f"Generated disulfides for {len(result)} classes.")
    else:
        print("Specify --class_ids or --all.")
        exit(1)

    if args.output and result:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved to {args.output}.")
