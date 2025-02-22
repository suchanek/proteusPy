"""
DisulfideBond Class Analysis Dictionary creation
Author: Eric G. Suchanek, PhD.
License: BSD
Last Modification: 2025-01-22 00:02:43 -egs-

Disulfide Class creation and manipulation. Binary classes using the +/- formalism of Hogg et al. 
(Biochem, 2006, 45, 7429-7433), are created for all 32 possible classes from the Disulfides 
extracted. Classes are named per Hogg's convention. This approach is extended to create 
eightfold classes based on the subdividing each dihedral angle chi1 - chi5 into 
8 equal segments, effectively quantizing them.
"""

# pylint: disable=C0301
# pylint: disable=C0103

import itertools
import pickle
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# from proteusPy.DisulfideBase import DisulfideList
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    SS_CLASS_DEFINITIONS,
    SS_CONSENSUS_BIN_FILE,
    SS_CONSENSUS_OCT_FILE,
)

_logger = create_logger(__name__)
_logger.setLevel("INFO")


class DisulfideClassManager:
    r"""
    This Class manages structural classes for the disulfide bonds contained
    in the proteusPy disulfide database.

    Class builds the internal dictionary mapping disulfides to class names.

    Disulfide binary classes are defined using the ± formalism described by
    Schmidt et al. (Biochem, 2006, 45, 7429-7433), across all 32 (2^5), possible
    binary sidechain torsional combinations. Classes are named per Schmidt's convention.
    The ``class_id`` represents the sign of each dihedral angle $\chi_{1} - \chi_{1'}$
    where *0* represents *negative* dihedral angle and *2* a *positive* angle.

    |   class_id | SS_Classname   | FXN        |   count |   incidence |   percentage |
    |-----------:|:---------------|:-----------|--------:|------------:|-------------:|
    |      00000 | -LHSpiral      | UNK        |   40943 |  0.23359    |    23.359    |
    |      00002 | 00002          | UNK        |    9391 |  0.0535781  |     5.35781  |
    |      00020 | -LHHook        | UNK        |    4844 |  0.0276363  |     2.76363  |
    |      00022 | 00022          | UNK        |    2426 |  0.0138409  |     1.38409  |
    |      00200 | -RHStaple      | Allosteric |   16146 |  0.092117   |     9.2117   |
    |      00202 | 00202          | UNK        |    1396 |  0.00796454 |     0.796454 |
    |      00220 | 00220          | UNK        |    7238 |  0.0412946  |     4.12946  |
    |      00222 | 00222          | UNK        |    6658 |  0.0379856  |     3.79856  |
    |      02000 | 02000          | UNK        |    7104 |  0.0405301  |     4.05301  |
    |      02002 | 02002          | UNK        |    8044 |  0.0458931  |     4.58931  |
    |      02020 | -LHStaple      | UNK        |    3154 |  0.0179944  |     1.79944  |
    |      02022 | 02022          | UNK        |    1146 |  0.00653822 |     0.653822 |
    |      02200 | -RHHook        | UNK        |    7115 |  0.0405929  |     4.05929  |
    |      02202 | 02202          | UNK        |    1021 |  0.00582507 |     0.582507 |
    |      02220 | -RHSpiral      | UNK        |    8989 |  0.0512845  |     5.12845  |
    |      02222 | 02222          | UNK        |    7641 |  0.0435939  |     4.35939  |
    |      20000 | ±LHSpiral      | UNK        |    5007 |  0.0285662  |     2.85662  |
    |      20002 | +LHSpiral      | UNK        |    1611 |  0.00919117 |     0.919117 |
    |      20020 | ±LHHook        | UNK        |    1258 |  0.00717721 |     0.717721 |
    |      20022 | +LHHook        | UNK        |     823 |  0.00469542 |     0.469542 |
    |      20200 | ±RHStaple      | UNK        |     745 |  0.00425042 |     0.425042 |
    |      20202 | +RHStaple      | UNK        |     538 |  0.00306943 |     0.306943 |
    |      20220 | ±RHHook        | Catalytic  |    1907 |  0.0108799  |     1.08799  |
    |      20222 | 20222          | UNK        |    1159 |  0.00661239 |     0.661239 |
    |      22000 | -/+LHHook      | UNK        |    3652 |  0.0208356  |     2.08356  |
    |      22002 | 22002          | UNK        |    2052 |  0.0117072  |     1.17072  |
    |      22020 | ±LHStaple      | UNK        |    1791 |  0.0102181  |     1.02181  |
    |      22022 | +LHStaple      | UNK        |     579 |  0.00330334 |     0.330334 |
    |      22200 | -/+RHHook      | UNK        |    8169 |  0.0466062  |     4.66062  |
    |      22202 | +RHHook        | UNK        |     895 |  0.0051062  |     0.51062  |
    |      22220 | ±RHSpiral      | UNK        |    3581 |  0.0204305  |     2.04305  |
    |      22222 | +RHSpiral      | UNK        |    8254 |  0.0470912  |     4.70912  |
    """

    def __init__(self, loader, verbose=True) -> None:
        self.verbose = verbose
        self.binaryclass_dict = {}
        self.binaryclass_df = None
        self.eightclass_df = None
        self.eightclass_dict = {}
        self.consensus_binary_list = None
        self.consensus_oct_list = None

        if self.verbose:
            _logger.info(
                "Loading binary consensus structure list from %s", SS_CONSENSUS_BIN_FILE
            )
        self.consensus_binary_list = self.load_consensus_file(octant=False)

        if self.verbose:
            _logger.info(
                "Loading octant consensus structure list from %s", SS_CONSENSUS_OCT_FILE
            )
        self.consensus_oct_list = self.load_consensus_file(octant=True)

        self.build_classes(loader)

    def __getitem__(self, item: str) -> np.ndarray:
        """
        Implements indexing against a class ID string.

        Return an array of disulfide IDs given the input Class ID string.

        :param item: The class ID string to index.
        :type item: str
        :return: An array of disulfide IDs corresponding to the class ID.
        :rtype: np.ndarray
        :raises ValueError: If an integer index is provided.
        :raises DisulfideException: If the class ID is invalid.
        """
        disulfides = None

        if isinstance(item, int):
            raise ValueError("Integer indexing not supported. Use a string key.")

        if isinstance(item, str):
            disulfides = self.class_to_sslist(item)
            return disulfides

        return disulfides

    def load_consensus_file(self, fpath=Path(DATA_DIR), octant=True):
        """Load the consensus file from the specified file."""

        res = None
        if octant:
            fname = fpath / SS_CONSENSUS_OCT_FILE
        else:
            fname = fpath / SS_CONSENSUS_BIN_FILE

        if not fname.exists():
            _logger.error("Cannot find file %s", fname)
            raise FileNotFoundError(f"Cannot find file {fname}")

        with open(fname, "rb") as f:
            res = pickle.load(f)
        return res

    def build_class_df(self, class_df, group_df):
        """Build a new DataFrame from the input DataFrames."""
        ss_id_col = group_df["ss_id"]
        result_df = pd.concat([class_df, ss_id_col], axis=1)
        return result_df

    def class_to_sslist(self, clsid: str, base=8) -> np.ndarray:
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

        if not isinstance(clsid, str):
            _logger.error("Invalid class ID: %s", clsid)
            return np.array([])

        match len(clsid):
            case 6:
                match clsid[-1]:
                    case "b":
                        eightorbin = self.binaryclass_dict
                    case "o":
                        eightorbin = self.eightclass_dict
                    case _:
                        _logger.error("Invalid class ID suffix: %s", clsid)
                        return np.array([])

            case 5:
                match base:
                    case 8:
                        eightorbin = self.eightclass_dict
                    case 2:
                        eightorbin = self.binaryclass_dict
                    case _:
                        _logger.error("Invalid base: %d", base)
                        return np.array([])
            case _:
                _logger.error("Invalid class ID length: %s", clsid)
                return np.array([])

        try:
            ss_ids = eightorbin[cls]

        except KeyError:
            _logger.error("Cannot find key %s in SSBond DB", clsid)
            return np.array([])

        return ss_ids

    def list_classes(self, base=2):
        """
        List the Disulfide structural classes.

        :param self: The instance of the DisulfideClassManager class.
        :type self: DisulfideClassManager
        :param base: The base class to use, 2 or 8.
        :type base: int
        :return: None
        :rtype: None
        :raises ValueError: If an invalid base value is provided.
        """
        match base:
            case 2:
                for k, v in enumerate(self.binaryclass_dict):
                    print(f"Class: |{k}|, |{v}|")
            case 8:
                for k, v in enumerate(self.eightclass_dict):
                    print(f"Class: |{k}|, |{v}|")
            case _:
                raise ValueError("Invalid base. Must be 2 or 8.")

    def concat_dataframes(self, df1, df2):
        """
        Concatenates columns from one data frame into the other
        and returns the new result.

        Parameters
        ----------
        df1 : pandas.DataFrame
            The first data frame.
        df2 : pandas.DataFrame
            The second data frame.

        Returns
        -------
        pandas.DataFrame
            The concatenated data frame.

        """
        # Merge the data frames based on the 'SS_Classname' column
        result = pd.merge(df1, df2, on="class_id")

        return result

    def binary_to_class(self, class_str: str, base: int = 8) -> list:
        """
        Convert a binary input string to a list of possible class strings based on the specified base.

        Returns a list of all possible combinations of ordinal sections of a unit circle
        divided into the specified number of equal segments, originating at 0 degrees, rotating counterclockwise,
        based on the sign of each angle in the input string.

        :param class_str: A string of length 5, where each character represents the sign
        of an angle in the range of -180-180 degrees.
        :type class_str: str
        :param base: The base class to use, 6 or 8.
        :type base: int
        :return: A list of strings of length 5, representing all possible class strings.
        :rtype: list
        :raises ValueError: If an invalid base value is provided.
        """
        match base:
            case 6:
                angle_maps = {"0": ["4", "5", "6"], "2": ["1", "2", "3"]}
            case 8:
                angle_maps = {"0": ["5", "6", "7", "8"], "2": ["1", "2", "3", "4"]}
            case 10:
                angle_maps = {
                    "0": ["6", "7", "8", "9", "A"],
                    "2": ["1", "2", "3", "4", "5"],
                }
            case _:
                raise ValueError("Invalid base value. Must be 6, 8, or 10.")

        class_lists = [angle_maps[char] for char in class_str]
        class_combinations = itertools.product(*class_lists)
        class_strings = ["".join(combination) for combination in class_combinations]
        return class_strings

    def build_classes(self, loader) -> None:
        """
        Build the internal structures needed for the binary and octant disulfide structural classes
        based on dihedral angle rules.

        :param loader: The DisulfideLoader object containing the data.
        :type loader: DisulfideLoader
        :return: None
        :rtype: None
        """

        from proteusPy import __version__

        self.version = __version__

        tors_df = loader.getTorsions()

        if self.verbose:
            _logger.info("Creating binary SS classes...")

        grouped = self.create_binary_classes(tors_df)

        class_df = pd.read_csv(
            StringIO(SS_CLASS_DEFINITIONS),
            dtype={
                "class_id": "string",
                "FXN": "string",
                "SS_Classname": "string",
            },
        )
        class_df["FXN"].str.strip()
        class_df["SS_Classname"].str.strip()
        class_df["class_id"].str.strip()

        merged = self.concat_dataframes(class_df, grouped)
        merged.drop(
            columns=["Idx", "chi1_s", "chi2_s", "chi3_s", "chi4_s", "chi5_s"],
            inplace=True,
        )

        classdict = dict(zip(merged["class_id"], merged["ss_id"]))
        self.binaryclass_dict = classdict
        self.binaryclass_df = merged.copy()

        if self.verbose:
            _logger.info("Creating eightfold SS classes...")

        grouped_eightclass = self.create_classes(tors_df, 8)
        self.eightclass_df = grouped_eightclass.copy()
        self.eightclass_dict = dict(
            zip(grouped_eightclass["class_id"], grouped_eightclass["ss_id"])
        )

        if self.verbose:
            _logger.info("Initialization complete.")

        return

    def create_binary_classes(self, df) -> pd.DataFrame:
        """
        Group the DataFrame by the sign of the chi columns and create a new class ID column for each unique grouping.

        :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance',
        'cb_distance', 'torsion_length', and 'energy'.
        :return: A pandas DataFrame containing columns 'class_id', 'ss_id', and 'count', where 'class_id'
         is a unique identifier for each grouping of chi signs, 'ss_id' is a list of all 'ss_id' values in that
         grouping, and 'count' is the number of rows in that grouping.
        """
        # Create new columns with the sign of each chi column
        chi_columns = ["chi1", "chi2", "chi3", "chi4", "chi5"]
        sign_columns = [col + "_s" for col in chi_columns]
        df[sign_columns] = df[chi_columns].map(lambda x: 1 if x >= 0 else -1)

        # Create a new column with the class ID for each row
        class_id_column = "class_id"
        df[class_id_column] = (df[sign_columns] + 1).apply(
            lambda x: "".join(x.astype(str)), axis=1
        )

        # Group the DataFrame by the class ID and return the grouped data
        grouped = df.groupby(class_id_column)["ss_id"].unique().reset_index()
        grouped["count"] = grouped["ss_id"].apply(len)
        grouped["incidence"] = grouped["count"] / len(df)
        grouped["percentage"] = grouped["incidence"] * 100

        return grouped

    def create_classes(self, df, base=8) -> pd.DataFrame:
        """
        Create a new DataFrame with 8-class encoding for input 'chi' values.

        The function takes a pandas DataFrame containing the following columns:
        'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance',
        'torsion_length', 'energy', and 'rho', and adds a class ID column based on the following rules:

        1. A new column named `class_id` is added, which is the concatenation of the individual class IDs per Chi.
        2. The DataFrame is grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
        the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and
        the percentage of incidence.

        :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
                'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
        :return: The grouped DataFrame with the added class column.
        """

        if base not in [2, 4, 6, 8, 10]:
            raise ValueError("Base must be either 2, 4, 6, 8, or 10")

        # Apply get_class_string function to each chi column based on the specified base
        for col_name in ["chi1", "chi2", "chi3", "chi4", "chi5"]:
            df[col_name + "_t"] = df[col_name].apply(
                lambda x: DisulfideClassManager.get_class_string(x, base=base)
            )

        # Concatenate the transformed chi columns to create class_id
        df["class_id"] = df[["chi1_t", "chi2_t", "chi3_t", "chi4_t", "chi5_t"]].agg(
            "".join, axis=1
        )

        # Group by class_id and aggregate required statistics
        grouped = df.groupby("class_id").agg({"ss_id": "unique"})
        grouped["count"] = grouped["ss_id"].str.len()
        grouped["incidence"] = grouped["count"] / len(df)
        grouped["percentage"] = grouped["incidence"] * 100

        return grouped.reset_index()

    def filter_class_by_percentage(self, cutoff: float, base: int = 8) -> pd.DataFrame:
        """
        Filter the specified class definitions by percentage.

        :param cutoff: A numeric value specifying the minimum percentage required for a row to be included in the output
        :param base: An optional integer specifying the class type to filter, defaults to 8
        :return: A new Pandas DataFrame containing only rows where the percentage is greater than or equal to the cutoff
        :rtype: pandas.DataFrame
        """

        match base:
            case 8:
                df = self.eightclass_df
            case 2:
                df = self.binaryclass_df
            case _:
                raise ValueError("Invalid base. Must be 6 or 8.")

        return df[df["percentage"] >= cutoff].copy()

    @staticmethod
    def get_binary_quadrant(angle_deg) -> str:
        """
        Return the binary quadrant in which an angle in degrees lies if the area is described by dividing a unit circle into 2 equal segments.

        :param angle_deg (float or array-like): The angle in degrees.

        Returns:
        :return str. The binary quadrant (0 or 2) that the angle belongs to.
        """
        angle_deg = (
            np.array(angle_deg) % 360
        )  # Normalize the angle to the range [0, 360)

        if np.isscalar(angle_deg):
            if angle_deg >= 0 and angle_deg < 180:
                return str(2)

            if angle_deg >= 180 and angle_deg < 360:
                return str(0)

            raise ValueError(
                "Invalid angle value: angle must be in the range [-360, 360)."
            )

        quadrants = np.where((angle_deg >= 0) & (angle_deg < 180), "2", "0")
        return "".join(quadrants)

    @staticmethod
    def get_sixth_quadrant(angle_deg) -> str:
        """
        Return the eighth quadrant in which an angle in degrees lies by dividing a unit circle into 8 equal segments.

        :param angle_deg: float or array-like
            The angle in degrees.
        :return: str
        """
        # Normalize the angle to the range [0, 360)
        angle_deg = np.array(angle_deg) % 360

        # Calculate quadrant: each quadrant spans 60 degrees.
        # The quadrant is computed as: quadrant = 6 - floor(angle_deg / 60)
        quadrant = 6 - np.floor_divide(angle_deg, 60).astype(int)

        # If the input was scalar (0-dim array), return a string.
        if angle_deg.ndim == 0:
            return str(quadrant.item())
        else:
            return "".join(quadrant.astype(str))

    @staticmethod
    def get_eighth_quadrant(angle_deg) -> str:
        """
        Return the eighth quadrant in which an angle in degrees lies by dividing a unit circle into 8 equal segments.

        :param angle_deg: float or array-like
            The angle in degrees.
        :return: str
        """

        # Normalize the angle to the range [0, 360)
        angle_deg = np.asarray(angle_deg) % 360

        # Calculate quadrant: each quadrant spans 45 degrees.
        # The quadrant is computed as: quadrant = 8 - floor(angle_deg / 45)
        quadrant = 8 - np.floor_divide(angle_deg, 45).astype(int)

        # If the input was scalar (0-dim array), return a string.
        if angle_deg.ndim == 0:
            return str(quadrant.item())

        return "".join(quadrant.astype(str))

    @staticmethod
    def get_class_string(*args, base=8) -> str:
        """
        Return the segment corresponding to an angle in degrees when the circle is divided into
        a specified number of equal segments. The segments are labeled in descending order, so that:

            - For base=8, an angle in [0, 45) degrees returns "8",
            an angle in [45, 90) returns "7", ...,
            and an angle in [315, 360) returns "1".

        :param angle_deg: The angle in degrees. Single or array-like with length 5.
        :type angle_deg: float or array-like
        :param base: The number of segments (default is 8). For example, with base=8 each segment spans 45°,
        with base=12 each spans 30°, etc.
        :type base: int, optional, default=8, max=16, must be even.
        :return: The label corresponding to the segment in which the input angle lies. For scalar input,
                    a single string is returned; for array-like input, a NumPy array of strings is returned.
        :rtype: str
        :raises ValueError: If the base is out of range.
        """
        # Character map for the segment labels, technically up to base 36.
        char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        angles = np.array(args).flatten()

        # Normalize the angle to [0, 360)
        angle_deg = np.asarray(angles) % 360

        # Each segment spans 360/base degrees.
        segment_size = 360 / base

        if len(args) not in [1, 5]:
            raise ValueError("You must enter either 1 or 5 dihedral angles.")

        if base > 16 or base < 2 and base % 2 != 0:
            raise ValueError("Invalid base. Must be even and between 2 and 16.")

        if base == 2:
            return DisulfideClassManager.get_binary_quadrant(angle_deg)

        # Compute the segment number: subtract the segment index (via floor division) from base.
        segment = base - np.floor_divide(angle_deg, segment_size).astype(int)

        # Return as string(s)
        if angle_deg.ndim == 0:
            return char_map[segment.item()]

        return "".join(char_map[i] for i in segment)

    @staticmethod
    def class_string_from_dihedral(*args, base=8) -> str:
        """
        Return the class string for a set of dihedral angles, given the base.

        :param args: One or five dihedral angles.
        :type args: float or array-like
        :param base: The base class to use, 2, 6, or 8. Defaults to 8.
        :type base: int, optional
        :return: The class string for the input dihedral angles.
        :rtype: str
        :raises ValueError: If the number of dihedral angles is not 1 or 5, or if the base is not 2, 6, or 8.
        """
        if len(args) not in [1, 5]:
            raise ValueError("You must enter either 1 or 5 dihedral angles.")

        if base > 16 or base < 2 and base % 2 != 0:
            raise ValueError("Invalid base. Must be between 2 and 16, and even.")

        angles = np.array(args).flatten()

        match len(angles):
            case 1:
                return DisulfideClassManager.get_class_string(angles[0], base=base)
            case 5:
                segments = [
                    DisulfideClassManager.get_class_string(angle, base=base)
                    for angle in angles
                ]
                return "".join(segments)
            case _:
                raise ValueError("Invalid number of angles. Must be 1 or 5.")

    def sslist_from_classid(self, cls: str, base=8) -> pd.DataFrame:
        """
        Return the 'ss_id' value in the given DataFrame that corresponds to the
        input 'cls' string in the class description
        """
        if base == 2:
            df = self.binaryclass_df
        elif base == 8:
            df = self.eightclass_df
        else:
            raise ValueError("Invalid base. Must be 2 or 8.")

        filtered_df = df[df["class_id"] == cls]

        if len(filtered_df) == 0:
            return None

        if len(filtered_df) > 1:
            raise ValueError(f"Multiple rows found for class_id '{cls}'")

        return filtered_df.iloc[0]["ss_id"]

    def class_to_binary(self, cls_str, base=8):
        """
        Return a string of length 5 representing the ordinal section of a unit circle for an angle in range -180-180 degrees
        into a string of 5 characters, where each character is either '0' if the corresponding input character represents a
        negative angle or '2' if it represents a positive angle.

        :param cls_str (str): A string of length 5 representing the ordinal section of a unit circle for an angle in range -180-180 degrees.
        :param base (int): The base of the ordinal section (6 or 8).
        :raises ValueError: If the base is not 6 or 8 or 10.
        :return str: A string of length 5, where each character is either '0' or '2', representing the sign of the corresponding input angle.
        """
        if base not in [6, 8, 10]:
            raise ValueError("Base must be either 6, 8, or 10")

        output_str = ""
        for char in cls_str:
            match base:
                case 6:
                    if char in ["1", "2", "3"]:
                        output_str += "2"
                    elif char in ["4", "5", "6"]:
                        output_str += "0"
                case 8:
                    if char in ["1", "2", "3", "4"]:
                        output_str += "2"
                    elif char in ["5", "6", "7", "8"]:
                        output_str += "0"
                case 10:
                    if char in ["1", "2", "3", "4", "5"]:
                        output_str += "2"
                    elif char in ["6", "7", "8", "9", "A"]:
                        output_str += "0"
                case _:
                    raise ValueError("Invalid base. Must be either 6, 8, or 10")
        return output_str

    def get_class_df(self, base=8):
        """
        Get the Disulfide structural classes DataFrame.

        :param base: The base class to use, either 2 or 8. Defaults to 8.
        :type base: int
        :return: A DataFrame containing the class_id, count, incidence, and percentage columns.
        :rtype: pandas.DataFrame
        :raises ValueError: If the base is not 2 or 8.
        """
        columns = ["class_id", "count", "incidence", "percentage"]
        match base:
            case 2:
                class_df = self.binaryclass_df
            case 8:
                class_df = self.eightclass_df
            case _:
                raise ValueError("Invalid base. Must be 2, or 8.")

        result_df = class_df[columns]
        return result_df


# class definition ends

# end of file
