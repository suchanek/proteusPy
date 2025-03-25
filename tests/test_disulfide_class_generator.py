"""
Unit tests for the DisulfideClassGenerator module in the proteusPy package.

This module contains pytest tests for the DisulfideClassGenerator class, which is responsible
for generating disulfide conformations for different structural classes based on CSV data.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-19
"""

# pylint: disable=W0613
# pylint: disable=W0621
# pylint: disable=W0212


import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator
from proteusPy.ProteusGlobals import (
    BINARY_CLASS_METRICS_FILE,
    DATA_DIR,
    OCTANT_CLASS_METRICS_FILE,
)


@pytest.fixture
def sample_csv_file():
    """
    Create a temporary CSV file with sample class metrics data for testing.

    :return: Path to the temporary CSV file
    :rtype: str
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        # Create sample data with required columns
        data = {
            "class": ["1", "2"],
            "class_str": ["+-+++", "-+---"],
            "chi1_mean": [-60.0, 60.0],
            "chi1_std": [10.0, 10.0],
            "chi2_mean": [-60.0, 60.0],
            "chi2_std": [10.0, 10.0],
            "chi3_mean": [-90.0, 90.0],
            "chi3_std": [10.0, 10.0],
            "chi4_mean": [-60.0, 60.0],
            "chi4_std": [10.0, 10.0],
            "chi5_mean": [-60.0, 60.0],
            "chi5_std": [10.0, 10.0],
        }
        df = pd.DataFrame(data)
        df.to_csv(tmp.name, index=False)
        return tmp.name


@pytest.fixture
def generator(sample_csv_file):
    """
    Create a DisulfideClassGenerator instance.

    :param sample_csv_file: Path to the sample CSV file
    :type sample_csv_file: str
    :return: DisulfideClassGenerator instance
    :rtype: DisulfideClassGenerator
    """
    generator = DisulfideClassGenerator(csv_file=sample_csv_file)
    return generator


class TestDisulfideClassGenerator:
    """Tests for the DisulfideClassGenerator class."""

    def test_init_with_csv_file(self, sample_csv_file):
        """Test initialization with a CSV file."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
        assert generator.df is not None
        assert len(generator.df) == 2
        assert "class" in generator.df.columns
        assert "class_str" in generator.df.columns
        assert not generator.binary_class_disulfides
        assert not generator.octant_class_disulfides

    def test_init_with_binary_and_octant_metrics(self):
        """Test initialization with binary and octant metrics files."""
        # Skip if binary metrics file doesn't exist
        binary_path = Path(DATA_DIR) / BINARY_CLASS_METRICS_FILE
        octant_path = Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE

        if not binary_path.exists() or not octant_path.exists():
            pytest.skip(f"Metrics files not found: {binary_path} or {octant_path}")

        generator = DisulfideClassGenerator()
        assert generator.df is not None
        assert generator.binary_df is not None
        assert generator.octant_df is not None

    def test_init_with_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Create a non-existent file path
        non_existent_file = "/tmp/non_existent_file_" + str(os.getpid()) + ".csv"

        # Ensure the file doesn't exist
        if os.path.exists(non_existent_file):
            os.unlink(non_existent_file)

        # Now try to initialize with this non-existent file
        with pytest.raises(FileNotFoundError):
            DisulfideClassGenerator(csv_file=non_existent_file)

    def test_validate_df_missing_columns(self, sample_csv_file):
        """Test validation of DataFrame with missing columns."""
        # Create a CSV with missing columns
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            data = {
                "class": ["1", "2"],
                "class_str": ["+-+++", "-+---"],
                # Missing chi1_mean, etc.
            }
            df = pd.DataFrame(data)
            df.to_csv(tmp.name, index=False)

            with pytest.raises(ValueError):
                DisulfideClassGenerator(csv_file=tmp.name)

            os.unlink(tmp.name)

    def test_load_csv(self, sample_csv_file):
        """Test loading a CSV file."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)

        # Create a new CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            data = {
                "class": ["3", "4"],
                "class_str": ["+++++", "-----"],
                "chi1_mean": [-60.0, 60.0],
                "chi1_std": [10.0, 10.0],
                "chi2_mean": [-60.0, 60.0],
                "chi2_std": [10.0, 10.0],
                "chi3_mean": [-90.0, 90.0],
                "chi3_std": [10.0, 10.0],
                "chi4_mean": [-60.0, 60.0],
                "chi4_std": [10.0, 10.0],
                "chi5_mean": [-60.0, 60.0],
                "chi5_std": [10.0, 10.0],
            }
            df = pd.DataFrame(data)
            df.to_csv(tmp.name, index=False)

            # Load the new CSV
            generator.load_csv(tmp.name)
            assert len(generator.df) == 2
            assert "3" in generator.df["class"].values
            assert "4" in generator.df["class"].values

            os.unlink(tmp.name)

    def test_parse_class_string(self):
        """Test the parse_class_string method."""
        # Test binary class strings
        assert DisulfideClassGenerator.parse_class_string("+-+++") == (2, "+-+++")
        assert DisulfideClassGenerator.parse_class_string("+-+++b") == (2, "+-+++")

        # Test octant class strings
        assert DisulfideClassGenerator.parse_class_string("12345") == (8, "12345")
        assert DisulfideClassGenerator.parse_class_string("12345o") == (8, "12345")

        # Test invalid class strings
        with pytest.raises(ValueError):
            DisulfideClassGenerator.parse_class_string("12345x")

        # Test non-string input
        with pytest.raises(ValueError):
            DisulfideClassGenerator.parse_class_string(12345)

    def test_generate_for_class(self, generator):
        """Test generating disulfides for a specific class."""
        # Set up the generator to use binary classes
        generator.base = 2
        generator.binary_df = generator.df

        # Test with valid class ID - use class_str that matches the sample data
        disulfide_list = generator.generate_for_class("+-+++")
        assert isinstance(disulfide_list, DisulfideList)
        assert len(disulfide_list) == 243  # 3^5 combinations

        # Verify the class is stored in binary_class_disulfides
        assert "+-+++" in generator.binary_class_disulfides

        # Test with invalid class ID
        result = generator.generate_for_class("invalid")
        assert result is None

        # Set up the generator to use octant classes
        generator.base = 8
        generator.octant_df = generator.df

        # Test with class column
        class_disulfide_list = generator.generate_for_class("1")
        assert isinstance(class_disulfide_list, DisulfideList)
        assert len(class_disulfide_list) == 243

    def test_generate_for_selected_classes(self, generator):
        """Test generating disulfides for selected classes."""
        # Set up the generator to use binary classes
        generator.base = 2
        generator.binary_df = generator.df

        # Test with valid class IDs
        class_ids = [
            "+-+++",
            "-+---",
            "+++++",
            "-----",
            "++-++",
            "--+--",
            "+-+-+",
            "-+-+-",
            "+--++",
            "-++--",
        ]
        result = generator.generate_for_selected_classes(class_ids)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "+-+++" in result
        assert "-+---" in result
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

        # Test with invalid class IDs
        result = generator.generate_for_selected_classes(["invalid"])
        assert result == {}

    def test_generate_for_all_classes(self, generator):
        """Test generating disulfides for all classes."""
        # Set up the generator to use binary classes
        generator.base = 2
        generator.binary_df = generator.df
        generator.octant_df = generator.df

        result = generator.generate_for_all_classes()
        assert isinstance(result, dict)
        assert len(result) == 4  # 2 binary classes + 2 octant classes
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_disulfides_for_class_internal(self, generator):
        """Test the _generate_disulfides_for_class internal method."""
        # Set up the generator to use binary classes
        generator.base = 2

        # Get the first row from the DataFrame
        row = generator.df.iloc[0]

        # Extract the needed values from the row
        class_id = row["class_str"]
        chi_means = tuple(row[f"chi{i}_mean"] for i in range(1, 6))
        chi_stds = tuple(row[f"chi{i}_std"] for i in range(1, 6))

        # Generate disulfides with the new method signature
        disulfide_list = generator._generate_disulfides_for_class(
            class_id, chi_means, chi_stds
        )

        # Verify the result
        assert isinstance(disulfide_list, DisulfideList)
        assert len(disulfide_list) == 243
        assert disulfide_list.pdb_id == f"Class_{class_id}"

        # Check the first disulfide
        first_disulfide = disulfide_list[0]
        assert isinstance(first_disulfide, Disulfide)
        assert first_disulfide.name.startswith(f"{row['class_str']}_comb")

        # Check that the torsions are correctly set
        # First combination should be (mean-std) for all angles
        expected_torsions = [
            row["chi1_mean"] - row["chi1_std"],
            row["chi2_mean"] - row["chi2_std"],
            row["chi3_mean"] - row["chi3_std"],
            row["chi4_mean"] - row["chi4_std"],
            row["chi5_mean"] - row["chi5_std"],
        ]
        assert np.allclose(first_disulfide.dihedrals, expected_torsions)

        # Set up the generator to use octant classes and test again
        generator.base = 8
        # For octant classes, we would use the 'class' column instead of 'class_str'
        class_id = row["class"]
        chi_means = tuple(row[f"chi{i}_mean"] for i in range(1, 6))
        chi_stds = tuple(row[f"chi{i}_std"] for i in range(1, 6))

        disulfide_list = generator._generate_disulfides_for_class(
            class_id, chi_means, chi_stds
        )
        first_disulfide = disulfide_list[0]
        assert first_disulfide.name.startswith(f"{class_id}_comb")

    def test_getitem(self, generator):
        """Test the __getitem__ method."""
        # Set up the generator with some test data
        generator.binary_class_disulfides = {"+-+++": DisulfideList([], "test_binary")}
        generator.octant_class_disulfides = {"12345": DisulfideList([], "test_octant")}

        # Test binary class
        result = generator["+-+++"]
        assert result.pdb_id == "test_binary"

        # Test octant class
        result = generator["12345"]
        assert result.pdb_id == "test_octant"

        # Test with suffix
        result = generator["+-+++b"]
        assert result.pdb_id == "test_binary"

        result = generator["12345o"]
        assert result.pdb_id == "test_octant"

        # Test with class that needs to be generated
        # Mock the generate_for_class method to return a test DisulfideList
        with patch.object(
            generator,
            "generate_for_class",
            return_value=DisulfideList([], "generated_class"),
        ):
            result = generator["new_class"]
            assert result.pdb_id == "generated_class"
            generator.generate_for_class.assert_called_once_with("new_class")

        # Test with invalid class that can't be generated
        with patch.object(generator, "generate_for_class", return_value=None):
            with pytest.raises(
                KeyError, match="Class invalid not found or could not be generated"
            ):
                generator["invalid"]

    def test_class_to_sslist(self, generator):
        """Test the class_to_sslist method."""
        # Set up the generator with some test data
        generator.binary_class_disulfides = {"+-+++": DisulfideList([], "test_binary")}
        generator.octant_class_disulfides = {"12345": DisulfideList([], "test_octant")}

        # Test binary class
        result = generator.class_to_sslist("+-+++", base=2)
        assert result.pdb_id == "test_binary"

        # Test octant class
        result = generator.class_to_sslist("12345", base=8)
        assert result.pdb_id == "test_octant"

        # Test with suffix
        result = generator.class_to_sslist("+-+++b")
        assert result.pdb_id == "test_binary"

        result = generator.class_to_sslist("12345o")
        assert result.pdb_id == "test_octant"

        # Test with invalid class - should raise KeyError
        with pytest.raises(KeyError, match="Class invalid not found"):
            generator.class_to_sslist("invalid", base=2)

    def test_display(self, generator):
        """Test the display method."""
        # Mock the DisulfideList.display_overlay method
        with patch(
            "proteusPy.DisulfideBase.DisulfideList.display_overlay"
        ) as mock_display:
            # Set up the generator with some test data
            generator.binary_class_disulfides = {
                "+-+++": DisulfideList([], "test_binary")
            }
            generator.octant_class_disulfides = {
                "12345": DisulfideList([], "test_octant")
            }

            # Test displaying a binary class
            generator.display("+-+++")
            mock_display.assert_called_once()
            mock_display.reset_mock()

            # Test displaying an octant class
            generator.display("12345")
            mock_display.assert_called_once()
            mock_display.reset_mock()

            # Test with parameters
            generator.display(
                "+-+++",
                screenshot=True,
                movie=True,
                fname="test.png",
                light="dark",
                winsize=(800, 600),
            )
            mock_display.assert_called_once_with(
                screenshot=True,
                movie=True,
                verbose=True,
                fname="test.png",
                winsize=(800, 600),
                light="dark",
                dpi=300,
            )

    def test_csv_file_not_loaded(self, sample_csv_file):
        """Test error when CSV file is not loaded."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
        generator.df = None

        with pytest.raises(ValueError):
            generator.generate_for_class("+-+++")

        with pytest.raises(ValueError):
            generator.generate_for_selected_classes(["+-+++"])


class TestDirectClassMethods:
    """Tests for using the DisulfideClassGenerator class methods directly."""

    def test_generate_for_all_classes(self, sample_csv_file):
        """Test the generate_for_all_classes method directly."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
        # Set binary_df and octant_df to the same dataframe for testing
        generator.binary_df = generator.df
        generator.octant_df = generator.df
        result = generator.generate_for_all_classes()
        assert isinstance(result, dict)
        assert len(result) == 4  # 2 binary classes + 2 octant classes
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_for_selected_classes(self, sample_csv_file):
        """Test the generate_for_selected_classes method directly."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
        # Set base to 8 for octant classes
        generator.base = 8
        generator.octant_df = generator.df
        class_ids = ["1", "2"]
        result = generator.generate_for_selected_classes(class_ids)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_for_class(self, sample_csv_file):
        """Test the generate_for_class method directly."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)

        # Set up the generator to use binary classes
        generator.base = 2
        generator.binary_df = generator.df

        # Test with binary class string (+-+++)
        result = generator.generate_for_class("+-+++")
        assert isinstance(result, DisulfideList)
        assert len(result) == 243

        # Set up the generator to use octant classes
        generator.base = 8
        generator.octant_df = generator.df

        # Test with octant class string (numeric)
        result = generator.generate_for_class("1")
        assert isinstance(result, DisulfideList)
        assert len(result) == 243

        # Test with invalid class string - now we expect None instead of ValueError
        result = generator.generate_for_class("invalid")
        assert result is None


if __name__ == "__main__":
    pytest.main()
