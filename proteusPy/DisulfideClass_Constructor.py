"""
DisulfideBond Class Analysis Dictionary creation
Author: Eric G. Suchanek, PhD.
License: BSD
Last Modification: 2025-01-05 20:37:38 -egs-

Disulfide Class creation and manipulation. Binary classes using the +/- formalism of Hogg et al. 
(Biochem, 2006, 45, 7429-7433), are created for all 32 possible classes from the Disulfides 
extracted. Classes are named per Hogg's convention. This approach is extended to create 
sixfold and eightfold classes based on the subdividing each dihedral angle chi1 - chi5 into 
6 and 8 equal segments, respectively.
"""

# pylint: disable=C0301
# pylint: disable=C0103

import itertools
import pickle
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteusPy.angle_annotation import AngleAnnotation
from proteusPy.DisulfideList import DisulfideList
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    CLASSOBJ_FNAME,
    DATA_DIR,
    DPI,
    PBAR_COLS,
    SS_CLASS_DEFINITIONS,
    SS_CLASS_DICT_FILE,
    SS_CONSENSUS_BIN_FILE,
    SS_CONSENSUS_OCT_FILE,
)

_logger = create_logger(__name__)
_logger.setLevel("INFO")

merge_cols = [
    "chi1_s",
    "chi2_s",
    "chi3_s",
    "chi4_s",
    "chi5_s",
    "class_id",
    "SS_Classname",
    "FXN",
    "count",
    "incidence",
    "percentage",
    "ca_distance_mean",
    "ca_distance_std",
    "torsion_length_mean",
    "torsion_length_std",
    "energy_mean",
    "energy_std",
    "ss_id",
]


# class DisulfideClass_Constructor:
class DisulfideClass_Constructor:
    r"""
    This Class manages structural classes for the disulfide bonds contained
    in the proteusPy disulfide database.

    Class builds the internal dictionary mapping disulfides to class names.

    Disulfide binary classes are defined using the ± formalism described by
    Schmidt et al. (Biochem, 2006, 45, 7429-7433), across all 32 (2^5), possible
    binary sidechain torsional combinations. Classes are named per Schmidt's convention.
    The ``class_id`` represents the sign of each dihedral angle $\chi_{1} - \chi_{1'}$
    where *0* represents *negative* dihedral angle and *2* a *positive* angle.

    |   class_id | SS_Classname   | FXN        |   count |   incidence |
    |-----------:|:---------------|:-----------|--------:|------------:|
    |      00000 | -LHSpiral      | UNK        |   31513 |  0.261092   |
    |      00002 | 00002          | UNK        |    5805 |  0.0480956  |
    |      00020 | -LHHook        | UNK        |    3413 |  0.0282774  |
    |      00022 | 00022          | UNK        |    1940 |  0.0160733  |
    |      00200 | -RHStaple      | Allosteric |   12735 |  0.105512   |
    |      00202 | 00202          | UNK        |     993 |  0.00822721 |
    |      00220 | 00220          | UNK        |    5674 |  0.0470103  |
    |      00222 | 00222          | UNK        |    5092 |  0.0421883  |
    |      02000 | 02000          | UNK        |    4749 |  0.0393465  |
    |      02002 | 02002          | UNK        |    3774 |  0.0312684  |
    |      02020 | -LHStaple      | UNK        |    1494 |  0.0123781  |
    |      02022 | 02022          | UNK        |     591 |  0.00489656 |
    |      02200 | -RHHook        | UNK        |    5090 |  0.0421717  |
    |      02202 | 02202          | UNK        |     533 |  0.00441602 |
    |      02220 | -RHSpiral      | UNK        |    6751 |  0.0559335  |
    |      02222 | 02222          | UNK        |    3474 |  0.0287828  |
    |      20000 | ±LHSpiral      | UNK        |    3847 |  0.0318732  |
    |      20002 | +LHSpiral      | UNK        |     875 |  0.00724956 |
    |      20020 | ±LHHook        | UNK        |     803 |  0.00665302 |
    |      20022 | +LHHook        | UNK        |     602 |  0.0049877  |
    |      20200 | ±RHStaple      | UNK        |     419 |  0.0034715  |
    |      20202 | +RHStaple      | UNK        |     293 |  0.00242757 |
    |      20220 | ±RHHook        | Catalytic  |    1435 |  0.0118893  |
    |      20222 | 20222          | UNK        |     488 |  0.00404318 |
    |      22000 | -/+LHHook      | UNK        |    2455 |  0.0203402  |
    |      22002 | 22002          | UNK        |    1027 |  0.00850891 |
    |      22020 | ±LHStaple      | UNK        |    1046 |  0.00866633 |
    |      22022 | +LHStaple      | UNK        |     300 |  0.00248556 |
    |      22200 | -/+RHHook      | UNK        |    6684 |  0.0553783  |
    |      22202 | +RHHook        | UNK        |     593 |  0.00491313 |
    |      22220 | ±RHSpiral      | UNK        |    2544 |  0.0210776  |
    |      22222 | +RHSpiral      | UNK        |    3665 |  0.0303653  |
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
        self.consensus_binary_list = self.load_consensus_file(oct=False)

        if self.verbose:
            _logger.info(
                "Loading octant consensus structure list from %s", SS_CONSENSUS_OCT_FILE
            )
        self.consensus_oct_list = self.load_consensus_file(oct=True)

        self.build_classes(loader)

    # overload __getitem__ to handle slicing and indexing, and access by name
    def __getitem__(self, item: str):
        """
        Implements indexing and slicing to retrieve DisulfideList objects from the
        DisulfideLoader. Supports:

        - Return a DisulfideList given the input Class ID string. Works with Binary
        classes.

        Raises DisulfideException on invalid indices or names.
        """
        disulfides = None

        if isinstance(item, int):
            raise ValueError("Integer indexing not supported. Use a string key.")

        elif isinstance(item, str):
            classdict = self.binaryclass_dict

            try:
                disulfides = classdict[item]
                return disulfides
            except KeyError:
                _logger.error(
                    "DisulfideLoader(): Cannot find key <%s> in SSBond DB", item
                )

    def load_class_dict(self, fname=Path(DATA_DIR) / SS_CLASS_DICT_FILE) -> dict:
        """Load the class dictionary from the specified file."""
        with open(fname, "rb") as f:
            self.binaryclass_dict = pickle.load(f)

    def load_consensus_file(self, fpath=Path(DATA_DIR), oct=True) -> DisulfideList:
        """Load the consensus file from the specified file."""

        res = None
        if oct:
            fname = fpath / SS_CONSENSUS_OCT_FILE
        else:
            fname = fpath / SS_CONSENSUS_BIN_FILE

        with open(fname, "rb") as f:
            res = pickle.load(f)
        return res

    def build_class_df(self, class_df, group_df):
        """Build a new DataFrame from the input DataFrames."""
        ss_id_col = group_df["ss_id"]
        result_df = pd.concat([class_df, ss_id_col], axis=1)
        return result_df

    def list_classes(self, base=2):
        """
        List the Disulfide structural classes.

        :param self: The instance of the DisulfideClass_Constructor class.
        :type self: DisulfideClass_Constructor
        :return: A list of disulfide structural classes.
        :rtype: list
        """
        if base == 2:
            for k, v in enumerate(self.binaryclass_dict):
                print(f"Class: |{k}|, |{v}|")

        elif base == 8:
            for k, v in enumerate(self.eightclass_dict):
                print(f"Class: |{k}|, |{v}|")
        else:
            raise ValueError("Invalid base. Must be 2, 6, or 8.")

    def from_class(self, loader, classid: str) -> DisulfideList:
        """
        Return a list of disulfides corresponding to the input BINARY class ID
        string.

        :param classid: Class ID, e.g. '00200'
        :return: DisulfideList of class members
        """

        res = DisulfideList([], classid)

        try:
            sslist = self.binaryclass_dict[classid]
            if self.verbose:
                pbar = tqdm(sslist, ncols=PBAR_COLS)
                for ssid in pbar:
                    res.append(loader[ssid])
                return res
            else:
                return DisulfideList([loader[ssid] for ssid in sslist], classid)
        except KeyError:
            _logger.error("No class: {classid}")
        return

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

    def binary_to_six_class(self, class_str):
        """
        Convert a binary input string to a list of possible six-class strings.

        Returns a list of all possible combinations of ordinal sections of a unit circle
        divided into 6 equal segments, originating at 0 degrees, rotating counterclockwise,
        based on the sign of each angle in the input string.

        :param angle_str (str): A string of length 5, where each character represents the sign
        of an angle in the range of -180-180 degrees.

        :return list: A list of strings of length 5, representing all possible six-class strings.
        """

        angle_maps = {"0": ["4", "5", "6"], "2": ["1", "2", "3"]}
        class_lists = [angle_maps[char] for char in class_str]
        class_combinations = itertools.product(*class_lists)
        class_strings = ["".join(combination) for combination in class_combinations]
        return class_strings

    def binary_to_eight_class(self, class_str):
        """
        Convert a binary input string to a list of possible eight-class strings.

        Returns a list of all possible combinations of ordinal sections of a unit circle
        divided into 6 equal segments, originating at 0 degrees, rotating counterclockwise,
        based on the sign of each angle in the input string.

        :param angle_str (str): A string of length 5, where each character represents the sign
        of an angle in the range of -180-180 degrees.

        :return list: A list of strings of length 5, representing all possible six-class strings.
        """
        angle_maps = {"0": ["5", "6", "7", "8"], "2": ["1", "2", "3", "4"]}
        class_lists = [angle_maps[char] for char in class_str]
        class_combinations = itertools.product(*class_lists)
        class_strings = ["".join(combination) for combination in class_combinations]
        return class_strings

    def build_classes(self, loader) -> None:
        """
        Build the internal structures needed for the binary and six-fold disulfide structural classes
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
        df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)

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
        Create a new DataFrame from the input with a 6-class encoding for input 'chi' values.

        The function takes a pandas DataFrame containing the following columns:
        'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance',
        'torsion_length', 'energy', and 'rho', and adds a class ID column based on the following rules:

        1. A new column named `class_id` is added, which is the concatenation of the individual class IDs per Chi.
        2. The DataFrame is grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
        the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
        percentage of incidence.

        :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
                'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
        :return: The grouped DataFrame with the added class column.
        """

        _df = pd.DataFrame()
        if base == 6:
            for col_name in ["chi1", "chi2", "chi3", "chi4", "chi5"]:
                _df[col_name + "_t"] = df[col_name].apply(self.get_sixth_quadrant)
        elif base == 8:
            for col_name in ["chi1", "chi2", "chi3", "chi4", "chi5"]:
                _df[col_name + "_t"] = df[col_name].apply(self.get_eighth_quadrant)
        else:
            raise ValueError("Base must be either 6 or 8")

        df["class_id"] = _df[["chi1_t", "chi2_t", "chi3_t", "chi4_t", "chi5_t"]].apply(
            lambda x: "".join(x), axis=1
        )

        grouped = df.groupby("class_id").agg({"ss_id": "unique"})
        grouped["count"] = grouped["ss_id"].apply(lambda x: len(x))
        grouped["incidence"] = grouped["count"] / len(df)
        grouped["percentage"] = grouped["incidence"] * 100
        grouped.reset_index(inplace=True)

        return grouped

    def create_eight_classes(self, df) -> pd.DataFrame:
        """
        Create a new DataFrame from the input with a 6-class encoding for input 'chi' values.

        The function takes a pandas DataFrame containing the following columns:
        'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance',
        'torsion_length', 'energy', and 'rho', and adds a class ID column based on the following rules:

        1. A new column named `class_id` is added, which is the concatenation of the individual class IDs per Chi.
        2. The DataFrame is grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
        the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
        percentage of incidence.

        :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
                'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
        :return: The grouped DataFrame with the added class column.
        """

        _df = pd.DataFrame()
        # create the chi_t columns for each chi column
        for col_name in ["chi1", "chi2", "chi3", "chi4", "chi5"]:
            _df[col_name + "_t"] = df[col_name].apply(self.get_eighth_quadrant)

        # create the class_id column
        df["class_id"] = _df[["chi1_t", "chi2_t", "chi3_t", "chi4_t", "chi5_t"]].apply(
            lambda x: "".join(x), axis=1
        )

        # group the DataFrame by class_id and return the grouped data
        grouped = df.groupby("class_id").agg({"ss_id": "unique"})
        grouped["count"] = grouped["ss_id"].apply(lambda x: len(x))
        grouped["incidence"] = grouped["count"] / len(df)
        grouped["percentage"] = grouped["incidence"] * 100
        grouped.reset_index(inplace=True)

        return grouped

    def filter_class_by_percentage(self, base: int, cutoff: float) -> pd.DataFrame:
        """
        Filter the specified class definitions by percentage.

        :param base: An integer specifying the class type to filter (6 or 8)
        :param cutoff: A numeric value specifying the minimum percentage required for a row to be included in the output
        :return: A new Pandas DataFrame containing only rows where the percentage is greater than or equal to the cutoff
        :rtype: pandas.DataFrame
        """
        if base == 8:
            df = self.eightclass_df
        else:
            raise ValueError("Invalid base. Must be 6 or 8.")

        return df[df["percentage"] >= cutoff].copy()

    def get_sixth_quadrant(self, angle_deg):
        """
        Return the sextant in which an angle in degrees lies if the area is described by dividing a unit circle into 6 equal segments.

        :param angle_deg (float): The angle in degrees.

        Returns:
        :return int: The sextant (1-6) that the angle belongs to.
        """
        # Normalize the angle to the range [0, 360)
        angle_deg = angle_deg % 360

        if angle_deg >= 0 and angle_deg < 60:
            return str(6)
        elif angle_deg >= 60 and angle_deg < 120:
            return str(5)
        elif angle_deg >= 120 and angle_deg < 180:
            return str(4)
        elif angle_deg >= 180 and angle_deg < 240:
            return str(3)
        elif angle_deg >= 240 and angle_deg < 300:
            return str(2)
        elif angle_deg >= 300 and angle_deg < 360:
            return str(1)
        else:
            raise ValueError(
                "Invalid angle value: angle must be in the range [-360, 360)."
            )

    def get_eighth_quadrant(self, angle_deg):
        """
        Return the octant in which an angle in degrees lies if the area is described by dividing a unit circle into 8 equal segments.

        :param angle_deg (float): The angle in degrees.

        Returns:
        :return str: The octant (1-8) that the angle belongs to.
        """
        # Normalize the angle to the range [0, 360)
        angle_deg = angle_deg % 360

        if angle_deg >= 0 and angle_deg < 45:
            return str(8)
        elif angle_deg >= 45 and angle_deg < 90:
            return str(7)
        elif angle_deg >= 90 and angle_deg < 135:
            return str(6)
        elif angle_deg >= 135 and angle_deg < 180:
            return str(5)
        elif angle_deg >= 180 and angle_deg < 225:
            return str(4)
        elif angle_deg >= 225 and angle_deg < 270:
            return str(3)
        elif angle_deg >= 270 and angle_deg < 315:
            return str(2)
        elif angle_deg >= 315 and angle_deg < 360:
            return str(1)
        else:
            raise ValueError(
                "Invalid angle value: angle must be in the range [-360, 360)."
            )

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

    def save(self, savepath=DATA_DIR) -> None:
        """
        Save a copy of the fully instantiated class to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        """
        _fname = None
        fname = CLASSOBJ_FNAME

        _fname = f"{savepath}{fname}"

        if self.verbose:
            _logger.info("Writing %s", _fname)

        with open(_fname, "wb+") as f:
            pickle.dump(self, f)

        if self.verbose:
            _logger.info("Done.")

    def class_to_binary(self, cls_str, base=8):
        """
        Return a string of length 5 representing the ordinal section of a unit circle for an angle in range -180-180 degrees
        into a string of 5 characters, where each character is either '0' if the corresponding input character represents a
        negative angle or '2' if it represents a positive angle.

        :param cls_str (str): A string of length 5 representing the ordinal section of a unit circle for an angle in range -180-180 degrees.
        :param base (int): The base of the ordinal section (6 or 8).
        :return str: A string of length 5, where each character is either '0' or '2', representing the sign of the corresponding input angle.
        """
        if base not in [6, 8]:
            raise ValueError("Base must be either 6 or 8")

        output_str = ""
        for char in cls_str:
            if base == 6:
                if char in ["1", "2", "3"]:
                    output_str += "2"
                elif char in ["4", "5", "6"]:
                    output_str += "0"
            elif base == 8:
                if char in ["1", "2", "3", "4"]:
                    output_str += "2"
                elif char in ["5", "6", "7", "8"]:
                    output_str += "0"
        return output_str

    def plot_class_chart(self, classes: int) -> None:
        """
        Create a Matplotlib pie chart with `classes` segments of equal size.

        This function returns a figure representing the angular layout of
        disulfide torsional classes for input `n` classes.

        Parameters:
            classes (int): The number of segments to create in the pie chart.

        Returns:
            None

        Example:
        >>> plot_class_chart(4)

        This will create a pie chart with 4 equal segments.
        """

        # Helper function to draw angle easily.
        def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
            vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            xy = np.c_[[length, 0], [0, 0], vec2 * length].T + np.array(pos)
            ax.plot(*xy.T, color=acol)
            return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

        # fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
        fig, ax1 = plt.subplots(sharex=True)

        # ax1, ax2 = fig.subplots(1, 2, sharey=True, sharex=True)

        fig.suptitle("SS Torsion Classes")
        fig.set_dpi(DPI)
        fig.set_size_inches(6.2, 6)

        fig.canvas.draw()  # Need to draw the figure to define renderer

        # Showcase different text positions.
        ax1.margins(y=0.4)
        ax1.set_title("textposition")
        _text = f"${360/classes}°$"
        kw = dict(size=75, unit="points", text=_text)

        plot_angle(ax1, (0, 0), 360 / classes, textposition="outside", **kw)

        # Create a list of segment values
        # !!!
        values = [1 for _ in range(classes)]

        # Create the pie chart
        # fig, ax = plt.subplots()
        wedges, _ = ax1.pie(
            values,
            startangle=0,
            counterclock=False,
            wedgeprops=dict(width=0.65),
        )

        # Set the chart title and size
        ax1.set_title(f"{classes}-Class Angular Layout")

        # Set the segment colors
        color_palette = plt.cm.get_cmap("tab20", classes)
        ax1.set_prop_cycle("color", [color_palette(i) for i in range(classes)])

        # Create the legend
        legend_labels = [f"Class {i+1}" for i in range(classes)]
        legend = ax1.legend(
            wedges,
            legend_labels,
            title="Classes",
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
        )

        # Set the legend fontsize
        plt.setp(legend.get_title(), fontsize="large")
        plt.setp(legend.get_texts(), fontsize="medium")

        # Show the chart
        fig.show()


# class definition ends

# end of file
