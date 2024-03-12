"""
DisulfideBond Class Analysis Dictionary creation
Author: Eric G. Suchanek, PhD.
License: BSD
Last Modification: 2/19/24 -egs-

Disulfide Class creation and manipulation using the +/- formalism of Hogg et al. (Biochem, 2006, 45, 7429-7433), 
across all 32 possible classes. Classes are named per Hogg's convention.
"""

# this workflow reads in the torsion database, groups it by torsions
# to create the classes merges with the master class spreadsheet, and saves the
# resulting dict to {DATA_DIR}PDB_SS_merged.csv

__pdoc__ = {"__all__": True}

import pickle
from io import StringIO

import pandas as pd
import tqdm
from Bio.PDB import *

import proteusPy
from proteusPy.data import (
    CLASSOBJ_FNAME,
    DATA_DIR,
    SS_CLASS_DEFINITIONS,
    SS_CLASS_DICT_FILE,
    SS_CONSENSUS_FILE,
)
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideList import DisulfideList
from proteusPy.ProteusGlobals import DPI

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


class DisulfideClass_Constructor:
    """
    This Class manages structural classes for the disulfide bonds contained
    in the proteusPy disulfide database.

    Build the internal dictionary mapping disulfides to class names.

    Disulfide binary classes are defined using the ± formalism described by
    Schmidt et al. (Biochem, 2006, 45, 7429-7433), across all 32 (2^5), possible
    binary sidechain torsional combinations. Classes are named per Schmidt's convention.
    The ``class_id`` represents the sign of each dihedral angle $\chi_{1} - \chi_{1'}$
    where *0* repreents *negative* dihedral angle and *2* a *positive* angle.

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
        self.classdict = {}
        self.classdf = None
        self.sixclass_df = None

        if self.verbose:
            print(f"-> DisulfideClass_Constructor(): Building SS classes...")
        self.build_yourself(loader)

    def load_class_dict(self, fname=f"{DATA_DIR}{SS_CLASS_DICT_FILE}") -> dict:
        with open(fname, "rb") as f:
            self.classdict = pd.compat.pickle_compat.load(f)
            # self.classdict = pickle.load(f)

    def load_consensus_file(self, fname=f"{DATA_DIR}{SS_CONSENSUS_FILE}"):
        with open(fname, "rb") as f:
            res = pd.compat.pickle_compat.load(f)
            # res = pickle.load(f)
            return res

    def build_class_df(self, class_df, group_df):
        ss_id_col = group_df["ss_id"]
        result_df = pd.concat([class_df, ss_id_col], axis=1)
        return result_df

    def list_binary_classes(self):
        for k, v in enumerate(self.classdict):
            print(f"Class: |{k}|, |{v}|")

    def from_class(self, classid: str) -> DisulfideList:
        """
        Return a list of disulfides corresponding to the input class ID
        string.

        :param classid: Class ID, e.g. '+RHStaple'
        :return: DisulfideList of class members
        """
        try:
            # Check if running in Jupyter
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
        except NameError:
            from tqdm import tqdm

        from proteusPy.ProteusGlobals import PBAR_COLS

        res = DisulfideList([], classid)

        try:
            sslist = self.classdict[classid]
            if self.verbose:
                pbar = tqdm(sslist, ncols=PBAR_COLS)
                for ssid in pbar:
                    res.append(self[ssid])
                return res
            else:
                return DisulfideList([self[ssid] for ssid in sslist], classid)
        except KeyError:

            print(f"No class: {classid}")
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
        import itertools

        angle_maps = {"0": ["4", "5", "6"], "2": ["1", "2", "3"]}
        class_lists = [angle_maps[char] for char in class_str]
        class_combinations = itertools.product(*class_lists)
        class_strings = ["".join(combination) for combination in class_combinations]
        return class_strings

    def build_yourself(self, loader) -> None:
        """
        Build the internal structures needed for the binary and six-fold disulfide structural classes
        based on dihedral angle rules.

        Parameters
        ----------
        loader: DisulfideLoader object

        Returns
        -------
        None
        """
        import proteusPy

        self.version = proteusPy.__version__

        tors_df = loader.getTorsions()

        if self.verbose:
            print(f"-> DisulfideClass_Constructor(): creating binary SS classes...")

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

        if self.verbose:
            print(f"-> DisulfideClass_Constructor(): merging...")

        merged = self.concat_dataframes(class_df, grouped)
        merged.drop(
            columns=["Idx", "chi1_s", "chi2_s", "chi3_s", "chi4_s", "chi5_s"],
            inplace=True,
        )

        classdict = dict(zip(merged["SS_Classname"], merged["ss_id"]))
        self.classdict = classdict
        self.classdf = merged.copy()

        if self.verbose:
            print(f"-> DisulfideClass_Constructor(): creating sixfold SS classes...")

        grouped_sixclass = self.create_six_classes(tors_df)

        self.sixclass_df = grouped_sixclass.copy()

        if self.verbose:
            print(f"-> DisulfideClass_Constructor(): initialization complete.")

        return

    def create_binary_classes(self, df) -> pd.DataFrame:
        """
        Group the DataFrame by the sign of the chi columns and create a new class ID column for each unique grouping.

        :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance',
        'cb_distance', 'torsion_length', and 'energy'.
        :return: A pandas DataFrame containing columns 'class_id', 'ss_id', and 'count', where 'class_id' is a unique identifier for each grouping of chi signs, 'ss_id' is a list of all 'ss_id' values in that grouping, and 'count'
        is the number of rows in that grouping.
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
        grouped["count"] = grouped["ss_id"].apply(lambda x: len(x))
        grouped["incidence"] = grouped["ss_id"].apply(lambda x: len(x) / len(df))
        grouped["percentage"] = grouped["incidence"].apply(lambda x: 100 * x)

        return grouped

    def create_six_classes(self, df) -> pd.DataFrame:
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
            _df[col_name + "_t"] = df[col_name].apply(self.get_sixth_quadrant)

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

    def filter_sixclass_by_percentage(self, cutoff) -> pd.DataFrame:
        """
        Filter the six-class definitions by percentage.

        :param df: A Pandas DataFrame with an 'percentage' column to filter by
        :param cutoff: A numeric value specifying the minimum percentage required for a row to be included in the output
        :return: A new Pandas DataFrame containing only rows where the percentage is greater than or equal to the cutoff
        :rtype: pandas.DataFrame
        """
        df = self.sixclass_df

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

    def sslist_from_classid(self, cls: str) -> DisulfideList:
        """
        Return the list of Disulfides from the classID string.

        :param cls: ClassID string
        """
        if "0" in cls:
            return self._ss_from_binary_classid(cls)
        else:
            return self._ss_from_sixclassid(cls)

    def _ss_from_sixclassid(self, cls: str) -> pd.DataFrame:
        """
        Return the 'ss_id' value in the given DataFrame that corresponds to the
        input 'cls' string in the sixfold class description.
        """

        df = self.sixclass_df

        filtered_df = df[df["class_id"] == cls]
        if len(filtered_df) == 0:
            return None
        elif len(filtered_df) > 1:
            raise ValueError(f"Multiple rows found for class_id '{cls}'")
        return filtered_df.iloc[0]["ss_id"]

    def _ss_from_binary_classid(self, cls: str) -> pd.DataFrame:
        """
        Return the 'ss_id' value in the given DataFrame that corresponds to the
        input 'cls' string in the binary class description.
        """

        df = self.classdf

        filtered_df = df[df["class_id"] == cls]
        if len(filtered_df) == 0:
            raise ValueError(f"No rows found for class_id '{cls}'")
        elif len(filtered_df) > 1:
            raise ValueError(f"Multiple rows found for class_id '{cls}'")
        return filtered_df.iloc[0]["ss_id"]

    def save(self, savepath=DATA_DIR) -> None:
        """
        Save a copy of the fully instantiated class to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        """
        self.version = proteusPy.__version__

        fname = CLASSOBJ_FNAME

        _fname = f"{savepath}{fname}"

        if self.verbose:
            print(f"-> DisulfideLoader.save(): Writing {_fname}... ")

        with open(_fname, "wb+") as f:
            pickle.dump(self, f)

        if self.verbose:
            print(f"-> DisulfideLoader.save(): Done.")

    def six_class_to_binary(self, cls_str):
        """
        Transforms a string of length 5 representing the ordinal section of a unit circle for an angle in range -180-180 degrees
        into a string of 5 characters, where each character is either '1' if the corresponding input character represents a
        negative angle or '2' if it represents a positive angle.

        :param cls_str (str): A string of length 5 representing the ordinal section of a unit circle for an angle in range -180-180 degrees.
        :return str: A string of length 5, where each character is either '0' or '2', representing the sign of the corresponding input angle.
        """
        output_str = ""
        for char in cls_str:
            if char in ["1", "2", "3"]:
                output_str += "2"
            elif char in ["4", "5", "6"]:
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
        import matplotlib.pyplot as plt
        import numpy as np

        from proteusPy.angle_annotation import AngleAnnotation

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

        am7 = plot_angle(ax1, (0, 0), 360 / classes, textposition="outside", **kw)

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
