"""
Unit tests for the DisulfideClassGenerator module in the proteusPy package.

This module contains pytest tests for the DisulfideClassGenerator class, which is responsible
for generating disulfide conformations for different structural classes based on CSV data.

Author: Eric G. Suchanek, PhD
Last revision: 2025-04-25 19:54:11
"""

# pylint: disable=W0613
# pylint: disable=W0621
# pylint: disable=W0212

import os
import shutil
import tempfile
import time
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
def temp_dir():
    """
    Create a temporary directory for test files.

    :return: Path to the temporary directory
    :rtype: Path
    """
    temp_dir_path = Path(tempfile.gettempdir()) / f"proteuspy_test_{os.getpid()}"
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    yield temp_dir_path
    # Clean up with retries
    retries = 3
    for _ in range(retries):
        try:
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir_path, ignore_errors=True)
            break
        except PermissionError:
            time.sleep(0.1)  # Brief delay to allow file handles to release


@pytest.fixture
def sample_csv_file(temp_dir):
    """
    Create a temporary CSV file with sample class metrics data for testing.

    :param temp_dir: Temporary directory for storing the CSV
    :type temp_dir: Path
    :return: Path to the temporary CSV file
    :rtype: str
    """
    csv_path = temp_dir / f"test_metrics_{os.urandom(8).hex()}.csv"
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
    df.to_csv(csv_path, index=False)
    yield str(csv_path)
    # File will be cleaned up by temp_dir fixture


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
        binary_path = Path(DATA_DIR) / BINARY_CLASS_METRICS_FILE
        octant_path = Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE

        if not binary_path.exists() or not octant_path.exists():
            pytest.skip(f"Metrics files not found: {binary_path} or {octant_path}")

        generator = DisulfideClassGenerator()
        assert generator.df is not None
        assert generator.binary_df is not None
        assert generator.octant_df is not None

    def test_init_with_invalid_params(self, temp_dir):
        """Test initialization with invalid parameters."""
        non_existent_file = temp_dir / f"non_existent_{os.urandom(8).hex()}.csv"
        with pytest.raises(FileNotFoundError):
            DisulfideClassGenerator(csv_file=str(non_existent_file))

    def test_validate_df_missing_columns(self, temp_dir):
        """Test validation of DataFrame with missing columns."""
        csv_path = temp_dir / f"invalid_metrics_{os.urandom(8).hex()}.csv"
        data = {
            "class": ["1", "2"],
            "class_str": ["+-+++", "-+---"],
            # Missing chi1_mean, etc.
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            DisulfideClassGenerator(csv_file=str(csv_path))

    def test_load_csv(self, generator, temp_dir):
        """Test loading a CSV file."""
        csv_path = temp_dir / f"new_metrics_{os.urandom(8).hex()}.csv"
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
        df.to_csv(csv_path, index=False)
        generator.load_csv(str(csv_path))
        assert len(generator.df) == 2
        assert "3" in generator.df["class"].values
        assert "4" in generator.df["class"].values

    def test_parse_class_string(self):
        """Test the parse_class_string method."""
        assert DisulfideClassGenerator.parse_class_string("+-+++") == (2, "+-+++")
        assert DisulfideClassGenerator.parse_class_string("+-+++b") == (2, "+-+++")
        assert DisulfideClassGenerator.parse_class_string("12345") == (8, "12345")
        assert DisulfideClassGenerator.parse_class_string("12345o") == (8, "12345")
        with pytest.raises(ValueError):
            DisulfideClassGenerator.parse_class_string("12345x")
        with pytest.raises(ValueError):
            DisulfideClassGenerator.parse_class_string(12345)

    def test_generate_for_class(self, generator):
        """Test generating disulfides for a specific class."""
        generator.base = 2
        generator.binary_df = generator.df
        disulfide_list = generator.generate_for_class("+-+++")
        assert isinstance(disulfide_list, DisulfideList)
        assert len(disulfide_list) == 243
        assert "+-+++" in generator.binary_class_disulfides
        result = generator.generate_for_class("invalid")
        assert result is None
        generator.base = 8
        generator.octant_df = generator.df
        class_disulfide_list = generator.generate_for_class("1")
        assert isinstance(class_disulfide_list, DisulfideList)
        assert len(class_disulfide_list) == 243

    def test_generate_for_selected_classes(self, generator):
        """Test generating disulfides for selected classes."""
        generator.base = 2
        generator.binary_df = generator.df
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
        result = generator.generate_for_selected_classes(["invalid"])
        assert result == {}

    def test_generate_for_all_classes(self, generator):
        """Test generating disulfides for all classes."""
        generator.base = 2
        generator.binary_df = generator.df
        generator.octant_df = generator.df
        result = generator.generate_for_all_classes()
        assert isinstance(result, dict)
        assert len(result) == 4
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_disulfides_for_class_internal(self, generator):
        """Test the _generate_disulfides_for_class internal method."""
        generator.base = 2
        row = generator.df.iloc[0]
        class_id = row["class_str"]
        chi_means = tuple(row[f"chi{i}_mean"] for i in range(1, 6))
        chi_stds = tuple(row[f"chi{i}_std"] for i in range(1, 6))
        disulfide_list = generator._generate_disulfides_for_class(
            class_id, chi_means, chi_stds
        )
        assert isinstance(disulfide_list, DisulfideList)
        assert len(disulfide_list) == 243
        assert disulfide_list.pdb_id == f"Class_{class_id}"
        first_disulfide = disulfide_list[0]
        assert isinstance(first_disulfide, Disulfide)
        assert first_disulfide.name.startswith(f"{row['class_str']}_comb")
        expected_torsions = [
            row["chi1_mean"] - row["chi1_std"],
            row["chi2_mean"] - row["chi2_std"],
            row["chi3_mean"] - row["chi3_std"],
            row["chi4_mean"] - row["chi4_std"],
            row["chi5_mean"] - row["chi5_std"],
        ]
        assert np.allclose(first_disulfide.dihedrals, expected_torsions)
        generator.base = 8
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
        generator.binary_class_disulfides = {"+-+++": DisulfideList([], "test_binary")}
        generator.octant_class_disulfides = {"12345": DisulfideList([], "test_octant")}
        result = generator["+-+++"]
        assert result.pdb_id == "test_binary"
        result = generator["12345"]
        assert result.pdb_id == "test_octant"
        result = generator["+-+++b"]
        assert result.pdb_id == "test_binary"
        result = generator["12345o"]
        assert result.pdb_id == "test_octant"
        with patch.object(
            generator,
            "generate_for_class",
            return_value=DisulfideList([], "generated_class"),
        ):
            result = generator["new_class"]
            assert result.pdb_id == "generated_class"
            generator.generate_for_class.assert_called_once_with("new_class")
        with patch.object(generator, "generate_for_class", return_value=None):
            with pytest.raises(
                KeyError, match="Class invalid not found or could not be generated"
            ):
                generator["invalid"]

    def test_class_to_sslist(self, generator):
        """Test the class_to_sslist method."""
        generator.binary_class_disulfides = {"+-+++": DisulfideList([], "test_binary")}
        generator.octant_class_disulfides = {"12345": DisulfideList([], "test_octant")}
        result = generator.class_to_sslist("+-+++", base=2)
        assert result.pdb_id == "test_binary"
        result = generator.class_to_sslist("12345", base=8)
        assert result.pdb_id == "test_octant"
        result = generator.class_to_sslist("+-+++b")
        assert result.pdb_id == "test_binary"
        result = generator.class_to_sslist("12345o")
        assert result.pdb_id == "test_octant"
        with pytest.raises(KeyError, match="Class invalid not found"):
            generator.class_to_sslist("invalid", base=2)

    def test_display(self, generator):
        """Test the display method."""
        with patch(
            "proteusPy.DisulfideBase.DisulfideList.display_overlay"
        ) as mock_display:
            generator.binary_class_disulfides = {
                "+-+++": DisulfideList([], "test_binary")
            }
            generator.octant_class_disulfides = {
                "12345": DisulfideList([], "test_octant")
            }
            generator.display("+-+++")
            mock_display.assert_called_once()
            mock_display.reset_mock()
            generator.display("12345")
            mock_display.assert_called_once()
            mock_display.reset_mock()
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
        generator.binary_df = generator.df
        generator.octant_df = generator.df
        result = generator.generate_for_all_classes()
        assert isinstance(result, dict)
        assert len(result) == 4
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_for_selected_classes(self, sample_csv_file):
        """Test the generate_for_selected_classes method directly."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
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
        generator.base = 2
        generator.binary_df = generator.df
        result = generator.generate_for_class("+-+++")
        assert isinstance(result, DisulfideList)
        assert len(result) == 243
        generator.base = 8
        generator.octant_df = generator.df
        result = generator.generate_for_class("1")
        assert isinstance(result, DisulfideList)
        assert len(result) == 243
        result = generator.generate_for_class("invalid")
        assert result is None


if __name__ == "__main__":
    pytest.main()
