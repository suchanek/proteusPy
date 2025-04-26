"""
Generate disulfide conformations for structural classes based on CSV data.

This module provides functionality to create disulfide conformations for different
structural classes using mean and standard deviation values for each dihedral angle.
For each class, it generates all combinations of dihedral angles (mean-std, mean, mean+std)
for each of the 5 dihedral angles, resulting in 3^5 = 243 combinations per class.

Author: Eric G. Suchanek, PhD
Last Modification: 2025-03-23
"""

# pylint: disable=C0103

import itertools
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import tqdm

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.DisulfideVisualization import DisulfideVisualization
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    BINARY_CLASS_METRICS_FILE,
    DATA_DIR,
    OCTANT_CLASS_METRICS_FILE,
)

# Constants
NUM_DIHEDRAL_ANGLES = 5
VALUES_PER_ANGLE = 3
TOTAL_COMBINATIONS = VALUES_PER_ANGLE**NUM_DIHEDRAL_ANGLES  # 243
BINARY_BASE = 2
OCTANT_BASE = 8
DIHEDRAL_COLUMNS = [
    f"chi{i}_{stat}"
    for i in range(1, NUM_DIHEDRAL_ANGLES + 1)
    for stat in ["mean", "std"]
]

# Logger setup
_logger = create_logger(__name__)
_logger.setLevel("INFO")


class DisulfideClassGenerator:
    """
    A class for generating disulfide conformations for structural classes based on
    mean and standard deviation values for dihedral angles.
    """

    @staticmethod
    def parse_class_string(class_str: str) -> Tuple[int, str]:
        """
        Parse a class string to determine its base and return the string without suffixes.

        :param class_str: The class string (e.g., "11111", "11111b", "+-+-+").
        :type class_str: str
        :return: Tuple of (base, clean_string) where base is 2 (binary) or 8 (octant).
        :rtype: Tuple[int, str]
        :raises ValueError: If the class string format is invalid.
        """
        if not isinstance(class_str, str):
            raise ValueError(f"Class string must be a string, got {type(class_str)}")

        base = OCTANT_BASE  # Default to octant
        clean_string = class_str

        if len(class_str) == 6:
            if class_str[-1] == "b":
                base = BINARY_BASE
                clean_string = class_str[:5]
            elif class_str[-1] == "o":
                base = OCTANT_BASE
                clean_string = class_str[:5]
            else:
                raise ValueError(f"Invalid suffix in class string: {class_str}")
        elif "+" in class_str or "-" in class_str:
            base = BINARY_BASE
            clean_string = class_str[:5]

        return base, clean_string

    def __init__(
        self,
        csv_file: str = None,
        verbose: bool = False,
        precalc: bool = False,
        csv_base: int = None,
    ):
        """
        Initialize the DisulfideClassGenerator.

        :param csv_file: Path to CSV file with class metrics (optional).
        :type csv_file: str
        :param verbose: Enable debug logging and progress bars.
        :type verbose: bool
        :param precalc: Pre-calculate disulfides for all classes.
        :type precalc: bool
        :param csv_base: Base for CSV data (2 or 8, optional).
        :type csv_base: int
        :raises ValueError: If no valid data source is provided.
        """
        self.verbose = verbose
        self.binary_class_disulfides: Dict[str, DisulfideList] = {}
        self.octant_class_disulfides: Dict[str, DisulfideList] = {}
        self.df = None
        self.binary_df = None
        self.octant_df = None
        self.base = None

        self._initialize_data(csv_file, csv_base)
        if precalc:
            self.generate_for_all_classes()

    def _initialize_data(self, csv_file: str = None, base: int = None) -> None:
        """
        Load data from CSV or pickle files.

        :param csv_file: Path to CSV file (optional).
        :type csv_file: str
        :param base: Base class for CSV data (2 or 8, optional).
        :type base: int
        :raises ValueError: If no valid data source is provided.
        """
        binary_path = Path(DATA_DIR) / BINARY_CLASS_METRICS_FILE
        octant_path = Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE

        if csv_file:
            self.load_csv(csv_file, base)
        else:
            if binary_path.exists():
                self.binary_df = pd.read_pickle(binary_path).astype(
                    {"class": str, "class_str": str}
                )
                self.df = self.binary_df
                if self.verbose:
                    _logger.info("Loaded binary metrics from %s", binary_path)
            if octant_path.exists():
                self.octant_df = pd.read_pickle(octant_path).astype(
                    {"class": str, "class_str": str}
                )
                # Prefer octant metrics if available
                if self.octant_df is not None:
                    self.df = self.octant_df
                    if self.verbose:
                        _logger.info("Loaded octant metrics from %s", octant_path)
            if self.df is None:
                raise ValueError("No valid data source provided.")

        self._validate_df()

    def _validate_df(self) -> None:
        """
        Validate DataFrame has required columns.

        :raises ValueError: If required columns are missing.
        """
        if self.df is not None:
            missing = [
                col
                for col in DIHEDRAL_COLUMNS + ["class", "class_str"]
                if col not in self.df.columns
            ]
            if missing:
                raise ValueError(f"DataFrame missing columns: {missing}")
            _logger.debug("DataFrame validated with columns: %s", list(self.df.columns))

    def load_csv(self, csv_file: str, base: int = None) -> "DisulfideClassGenerator":
        """
        Load class metrics from a CSV file.

        :param csv_file: Path to CSV file.
        :type csv_file: str
        :param base: Base class (2 or 8, optional).
        :type base: int
        :return: Self for method chaining.
        :rtype: DisulfideClassGenerator
        :raises ValueError: If base is invalid.
        """
        self.df = pd.read_csv(csv_file, dtype={"class": str, "class_str": str})
        self.base = base
        if base == BINARY_BASE:
            self.binary_df = self.df
        elif base == OCTANT_BASE:
            self.octant_df = self.df
        elif base is not None:
            raise ValueError(f"Invalid base: {base}")
        return self

    def __getitem__(self, classid: str) -> DisulfideList:
        """
        Get a DisulfideList for a given class ID, generating it if it doesn't exist.

        This method allows for dictionary-like access to class disulfides using
        the [] operator, e.g., generator["11111"] or generator["+-+++b"].

        :param classid: Class ID (e.g., "11111", "11111b", "+-+++").
        :type classid: str
        :return: DisulfideList for the class.
        :rtype: DisulfideList
        :raises ValueError: If classid is invalid.
        :raises KeyError: If class not found and cannot be generated.
        """
        if not isinstance(classid, str):
            _logger.error("Class ID must be a string, got %s", type(classid))
            raise ValueError(f"Invalid class ID type: {type(classid)}")

        parsed_base, clean_cls = self.parse_class_string(classid)
        target_dict = (
            self.binary_class_disulfides
            if parsed_base == BINARY_BASE
            else self.octant_class_disulfides
        )

        # If the class is already generated, return it
        if clean_cls in target_dict:
            return target_dict[clean_cls]

        # Otherwise, try to generate it
        sslist = self.generate_for_class(classid)
        if sslist is None:
            _logger.error("Class %s not found or could not be generated", classid)
            raise KeyError(f"Class {classid} not found or could not be generated")

        return sslist

    def class_to_sslist(self, clsid: str, base: int = OCTANT_BASE) -> DisulfideList:
        """
        Retrieve disulfide list for a given class ID.

        :param clsid: Class ID (e.g., "11111", "11111b").
        :type clsid: str
        :param base: Base class (2 or 8).
        :type base: int
        :return: DisulfideList for the class.
        :rtype: DisulfideList
        :raises ValueError: If clsid is invalid.
        :raises KeyError: If class not found.
        """
        if not isinstance(clsid, str):
            _logger.error("Class ID must be a string, got %s", type(clsid))
            raise ValueError(f"Invalid class ID type: {type(clsid)}")

        parsed_base, clean_cls = self.parse_class_string(clsid)
        base = parsed_base if parsed_base else base
        target_dict = (
            self.binary_class_disulfides
            if base == BINARY_BASE
            else self.octant_class_disulfides
        )

        if clean_cls not in target_dict:
            _logger.error(
                "Class %s not found in %s disulfides",
                clsid,
                "binary" if base == BINARY_BASE else "octant",
            )
            raise KeyError(f"Class {clsid} not found")

        return target_dict[clean_cls]

    def _generate_disulfides_for_class(
        self, class_id: str, chi_means: tuple, chi_stds: tuple
    ) -> DisulfideList:
        """
        Generate disulfide conformations for a class (cached).

        :param class_id: Class identifier.
        :type class_id: str
        :param chi_means: Tuple of mean dihedral angles.
        :type chi_means: tuple
        :param chi_stds: Tuple of standard deviations.
        :type chi_stds: tuple
        :return: DisulfideList with all combinations.
        :rtype: DisulfideList
        """
        chi_values = [[m - s, m, m + s] for m, s in zip(chi_means, chi_stds)]
        combinations = list(itertools.product(*chi_values))
        disulfides = [
            Disulfide(f"{class_id}_comb{i+1}", torsions=list(combo))
            for i, combo in enumerate(combinations)
        ]
        return DisulfideList(disulfides, f"Class_{class_id}")

    def generate_for_class(self, class_id: str) -> Union[DisulfideList, None]:
        """
        Generate disulfides for a specific class.

        :param class_id: Class ID (e.g., "11111b").
        :type class_id: str
        :return: DisulfideList or None if class not found.
        :rtype: Union[DisulfideList, None]
        :raises ValueError: If no DataFrame is loaded.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        base, clean_string = self.parse_class_string(class_id)
        df_to_use = self.binary_df if base == BINARY_BASE else self.octant_df

        if df_to_use is None:
            _logger.warning(
                "%s DataFrame not available.",
                "Binary" if base == BINARY_BASE else "Octant",
            )
            return None

        col = "class_str" if base == BINARY_BASE else "class"
        row = df_to_use[df_to_use[col] == clean_string]
        if row.empty:
            _logger.warning("Class %s not found.", class_id)
            return None

        row_data = row.iloc[0]
        chi_means = tuple(
            row_data[f"chi{i}_mean"] for i in range(1, NUM_DIHEDRAL_ANGLES + 1)
        )
        chi_stds = tuple(
            row_data[f"chi{i}_std"] for i in range(1, NUM_DIHEDRAL_ANGLES + 1)
        )
        disulfide_list = self._generate_disulfides_for_class(
            clean_string, chi_means, chi_stds
        )

        target_dict = (
            self.binary_class_disulfides
            if base == BINARY_BASE
            else self.octant_class_disulfides
        )
        target_dict[clean_string] = disulfide_list
        _logger.info(
            "Generated %d disulfides for class %s", len(disulfide_list), class_id
        )
        return disulfide_list

    def generate_for_selected_classes(
        self, class_ids: List[str]
    ) -> Dict[str, DisulfideList]:
        """
        Generate disulfides for multiple classes in parallel.

        :param class_ids: List of class IDs.
        :type class_ids: List[str]
        :return: Dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If no DataFrame is loaded.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        class_disulfides = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.generate_for_class, cid): cid for cid in class_ids
            }
            for future in tqdm.tqdm(
                futures, desc="Generating classes", disable=not self.verbose
            ):
                class_id = futures[future]
                if (_result := future.result()) is not None:
                    class_disulfides[class_id] = _result

        _logger.info("Generated disulfides for %d classes", len(class_disulfides))
        return class_disulfides

    def generate_for_all_classes(self) -> Dict[str, DisulfideList]:
        """
        Generate disulfides for all classes in loaded DataFrames.

        :return: Dictionary mapping class IDs to DisulfideLists.
        :rtype: Dict[str, DisulfideList]
        :raises ValueError: If no DataFrames are loaded.
        """
        if self.binary_df is None and self.octant_df is None:
            raise ValueError("No DataFrames loaded.")

        class_disulfides = {}
        if self.binary_df is not None:
            self.base = BINARY_BASE
            for _, row in self._iterate_df(self.binary_df, "binary"):
                class_id = row["class_str"]
                disulfide_list = self._generate_from_row(row, class_id)
                class_disulfides[f"{class_id}b"] = disulfide_list
                self.binary_class_disulfides[class_id] = disulfide_list

        if self.octant_df is not None:
            self.base = OCTANT_BASE
            for _, row in self._iterate_df(self.octant_df, "octant"):
                class_id = row["class"]
                disulfide_list = self._generate_from_row(row, class_id)
                class_disulfides[f"{class_id}o"] = disulfide_list
                self.octant_class_disulfides[class_id] = disulfide_list

        _logger.info("Generated disulfides for %d classes", len(class_disulfides))
        return class_disulfides

    def _iterate_df(self, df: pd.DataFrame, desc: str):
        """
        Helper to iterate DataFrame with optional progress bar.

        :param df: DataFrame to iterate.
        :type df: pd.DataFrame
        :param desc: Description for progress bar.
        :type desc: str
        :return: Iterator over DataFrame rows.
        """
        return (
            tqdm.tqdm(
                df.iterrows(), total=len(df), desc=f"Generating {desc} disulfides"
            )
            if self.verbose
            else df.iterrows()
        )

    def _generate_from_row(self, row: pd.Series, class_id: str) -> DisulfideList:
        """
        Generate disulfides from a DataFrame row.

        :param row: DataFrame row with class metrics.
        :type row: pd.Series
        :param class_id: Class identifier.
        :type class_id: str
        :return: DisulfideList for the row.
        :rtype: DisulfideList
        """
        chi_means = tuple(
            row[f"chi{i}_mean"] for i in range(1, NUM_DIHEDRAL_ANGLES + 1)
        )
        chi_stds = tuple(row[f"chi{i}_std"] for i in range(1, NUM_DIHEDRAL_ANGLES + 1))
        return self._generate_disulfides_for_class(class_id, chi_means, chi_stds)

    def describe(self, detailed: bool = False) -> None:
        """
        Print the internal state of the generator.

        :param detailed: Show detailed DataFrame and disulfide info.
        :type detailed: bool
        """
        print("\n" + "=" * 80)
        print(f"{'DisulfideClassGenerator State':^80}")
        print("=" * 80)
        print(f"\nBase: {self.base or 'Not set'}")
        print(f"Verbose: {self.verbose}")

        print("\nDataFrames:")
        for name, df, attr in [
            ("Binary", self.binary_df, "binary_df"),
            ("Octant", self.octant_df, "octant_df"),
        ]:
            if df is not None:
                print(f"  {name}: Loaded ({len(df)} classes)")
                if detailed:
                    print(f"    Sample:\n{df.head(3)}")
            else:
                print(f"  {name}: Not loaded")

        print("\nDisulfides:")
        for name, d in [
            ("Binary", self.binary_class_disulfides),
            ("Octant", self.octant_class_disulfides),
        ]:
            count = len(d)
            print(f"  {name}: {count} classes")
            if detailed and count:
                print(f"    Sample: {list(d.items())[:3]}")
        print("=" * 80)

    def prepare_energy_data(self) -> pd.DataFrame:
        """
        Prepare energy data for visualization.

        :return: DataFrame with class and energy data.
        :rtype: pd.DataFrame
        """
        energy_data = []
        for base, d in [
            (BINARY_BASE, self.binary_class_disulfides),
            (OCTANT_BASE, self.octant_class_disulfides),
        ]:
            for class_id, disulfide_list in d.items():
                for ss in disulfide_list:
                    energy_data.append(
                        {"class": class_id, "energy": ss.energy, "base": base}
                    )
        df = pd.DataFrame(energy_data)
        (
            _logger.info("Prepared energy data with %d entries", len(df))
            if not df.empty
            else _logger.warning("No energy data.")
        )
        return df

    def plot_energy_by_class(
        self, base: int = None, title: str = "Energy Distribution by Class", **kwargs
    ) -> None:
        """
        Plot energy distribution by class.

        :param base: Filter by base (2 or 8, optional).
        :type base: int
        :param title: Plot title.
        :type title: str
        :param kwargs: Additional plotting options.
        """

        energy_df = self.prepare_energy_data()
        if energy_df.empty:
            _logger.warning("No energy data available.")
            return

        if base in (BINARY_BASE, OCTANT_BASE):
            energy_df = energy_df[energy_df["base"] == base]
            title = f"{'Binary' if base == BINARY_BASE else 'Octant'} {title}"
        elif base is not None:
            raise ValueError("Base must be 2 or 8.")

        DisulfideVisualization.plot_energy_by_class(energy_df, title=title, **kwargs)

    def plot_torsion_distance_by_class(
        self,
        base: int = None,
        title: str = "Torsion Distance by Class",
        theme: str = "auto",
        save: bool = False,
        savedir: str = ".",
        verbose: bool = False,
        split: bool = False,
        max_classes_per_plot: int = 85,
        dpi: int = 300,
        suffix: str = "png",
    ) -> None:
        """
        Create a bar chart showing torsion distance by class_id.

        :param base: The base class to use (2 for binary, 8 for octant). If None,
        use all available data.
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
        :param split: Whether to split the plot into multiple plots if there are many classes
        :type split: bool
        :param max_classes_per_plot: Maximum number of classes to include in each plot
        when splitting
        :type max_classes_per_plot: int
        :param dpi: DPI (dots per inch) for the saved image, controls the resolution (default: 600)
        :type dpi: int
        :param suffix: File format for saved images (default: "png")
        :type suffix: str
        """

        # Determine which DataFrame to use based on base parameter
        if base is not None:
            if base not in [2, 8]:
                raise ValueError("Base must be 2 (binary) or 8 (octant)")

            df_to_use = self.binary_df if base == 2 else self.octant_df
            base_str = "Binary" if base == 2 else "Octant"
            plot_title = f"{base_str} {title}"
        else:
            # Default to octant if available, otherwise binary
            df_to_use = self.octant_df if self.octant_df is not None else self.binary_df
            plot_title = title

        if df_to_use is None or df_to_use.empty:
            _logger.warning("No data available for the specified base.")
            return

        # Check if avg_torsion_distance column exists
        if "avg_torsion_distance" not in df_to_use.columns:
            _logger.warning("avg_torsion_distance column not found in the DataFrame.")
            return

        # Create a copy of the DataFrame to avoid modifying the original
        plot_df = df_to_use.copy()

        # If not splitting or few classes, create a single plot
        if not split or len(plot_df) <= max_classes_per_plot:
            # Create line plot using plotly_express
            DisulfideVisualization.plot_torsion_distance_by_class(
                plot_df,
                title=plot_title,
                theme=theme,
                save=save,
                savedir=savedir,
                verbose=verbose,
                dpi=dpi,
                suffix=suffix,
            )
        else:
            # Split into multiple plots
            num_plots = (
                len(plot_df) + max_classes_per_plot - 1
            ) // max_classes_per_plot

            if verbose:
                _logger.info(
                    "Splitting into %d plots with up to %d classes each",
                    num_plots,
                    max_classes_per_plot,
                )

            for i in range(num_plots):
                start_idx = i * max_classes_per_plot
                end_idx = min((i + 1) * max_classes_per_plot, len(plot_df))
                subset_df = plot_df.iloc[start_idx:end_idx]

                # Create plot title for this subset
                subset_title = f"{plot_title} (Part {i+1} of {num_plots})"
                fname_prefix = f"{plot_title.lower().replace(' ', '_')}_part_{i+1}"
                # Create line plot for this subset
                DisulfideVisualization.plot_torsion_distance_by_class(
                    subset_df,
                    title=subset_title,
                    theme=theme,
                    save=save,
                    savedir=savedir,
                    verbose=verbose,
                    dpi=dpi,
                    suffix=suffix,
                    fname_prefix=fname_prefix,
                )

    def display(
        self,
        class_id: str,
        screenshot: bool = False,
        movie: bool = False,
        fname: str = "ss_overlay.png",
        dpi: int = 300,
        winsize: tuple = (800, 600),
        light: str = "auto",
        **kwargs,
    ) -> None:
        """
        Display disulfide overlay for a class.

        :param class_id: Class ID to display.
        :type class_id: str
        :param screenshot: Save a screenshot.
        :type screenshot: bool
        :param movie: Save a movie.
        :type movie: bool
        :param fname: Output filename.
        :type fname: str
        :param dpi: DPI for the output image.
        :type dpi: int
        :param winsize: Window size for the display.
        :type winsize: tuple
        :param light: Light source for the display.
        :type light: str
        :param kwargs: Additional display options.
        :raises ValueError: If class not found or generation fails.
        """
        base, clean_string = self.parse_class_string(class_id)
        target_dict = (
            self.binary_class_disulfides
            if base == BINARY_BASE
            else self.octant_class_disulfides
        )

        if clean_string in target_dict:
            disulfide_list = target_dict[clean_string]
        else:
            disulfide_list = self.generate_for_class(class_id)
            if disulfide_list is None:
                raise ValueError(f"Class {class_id} not found or failed to generate.")

        disulfide_list.display_overlay(
            screenshot=screenshot,
            movie=movie,
            verbose=True,
            fname=fname,
            winsize=winsize,
            light=light,
            dpi=dpi,
            **kwargs,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate disulfide conformations.")
    parser.add_argument("csv_file", help="Path to CSV file.")
    parser.add_argument("--class_ids", nargs="+", help="Class IDs to process.")
    parser.add_argument("--all", action="store_true", help="Process all classes.")
    parser.add_argument("--output", help="Output pickle file.")
    parser.add_argument("--verbose", action="store_true", help="Show progress.")
    args = parser.parse_args()

    generator = DisulfideClassGenerator(args.csv_file, verbose=args.verbose)
    result = (
        generator.generate_for_all_classes()
        if args.all
        else generator.generate_for_selected_classes(args.class_ids or [])
    )

    if not result:
        print("Specify --class_ids or --all.")
        sys.exit(1)

    print(f"Generated disulfides for {len(result)} classes.")
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved to {args.output}.")

    # EOF
