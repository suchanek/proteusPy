"""
Unit tests for the DisulfideClassGenerator module in the proteusPy package.

This module contains pytest tests for the DisulfideClassGenerator class, which is responsible
for generating disulfide conformations for different structural classes based on CSV data.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-17
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.DisulfideClassGenerator import (
    DisulfideClassGenerator,
    generate_disulfides_for_all_classes,
    generate_disulfides_for_class_from_csv,
    generate_disulfides_for_selected_classes,
)
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
def binary_generator(sample_csv_file):
    """
    Create a DisulfideClassGenerator instance with binary base.
    
    :param sample_csv_file: Path to the sample CSV file
    :type sample_csv_file: str
    :return: DisulfideClassGenerator instance
    :rtype: DisulfideClassGenerator
    """
    return DisulfideClassGenerator(csv_file=sample_csv_file, base=2)


@pytest.fixture
def octant_generator(sample_csv_file):
    """
    Create a DisulfideClassGenerator instance with octant base.
    
    :param sample_csv_file: Path to the sample CSV file
    :type sample_csv_file: str
    :return: DisulfideClassGenerator instance
    :rtype: DisulfideClassGenerator
    """
    return DisulfideClassGenerator(csv_file=sample_csv_file, base=8)


class TestDisulfideClassGenerator:
    """Tests for the DisulfideClassGenerator class."""

    def test_init_with_csv_file(self, sample_csv_file):
        """Test initialization with a CSV file."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
        assert generator.df is not None
        assert len(generator.df) == 2
        assert "class" in generator.df.columns
        assert "class_str" in generator.df.columns
        assert generator.binary_class_disulfides == {}
        assert generator.octant_class_disulfides == {}

    def test_init_with_binary_base(self):
        """Test initialization with binary base."""
        # Skip if binary metrics file doesn't exist
        binary_path = Path(DATA_DIR) / BINARY_CLASS_METRICS_FILE
        if not binary_path.exists():
            pytest.skip(f"Binary metrics file not found: {binary_path}")
        
        generator = DisulfideClassGenerator(base=2)
        assert generator.df is not None
        assert generator.base == 2
        assert generator.binary_class_disulfides == {}

    def test_init_with_octant_base(self):
        """Test initialization with octant base."""
        # Skip if octant metrics file doesn't exist
        octant_path = Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE
        if not octant_path.exists():
            pytest.skip(f"Octant metrics file not found: {octant_path}")
        
        generator = DisulfideClassGenerator(base=8)
        assert generator.df is not None
        assert generator.base == 8
        assert generator.octant_class_disulfides == {}

    def test_init_with_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            DisulfideClassGenerator()  # No parameters provided
        
        with pytest.raises(ValueError):
            DisulfideClassGenerator(base=3)  # Invalid base

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

    def test_generate_for_class(self, binary_generator):
        """Test generating disulfides for a specific class."""
        # Test with valid class ID
        disulfide_list = binary_generator.generate_for_class("+-+++", use_class_str=True)
        assert isinstance(disulfide_list, DisulfideList)
        assert len(disulfide_list) == 243  # 3^5 combinations
        assert disulfide_list.pdb_id == "Class_1_+-+++"
        
        # Verify the class is stored in binary_class_disulfides
        assert "+-+++" in binary_generator.binary_class_disulfides
        
        # Test with invalid class ID
        result = binary_generator.generate_for_class("invalid", use_class_str=True)
        assert result is None
        
        # Test with class column
        class_disulfide_list = binary_generator.generate_for_class("1", use_class_str=False)
        assert isinstance(class_disulfide_list, DisulfideList)
        assert len(class_disulfide_list) == 243

    def test_generate_for_selected_classes(self, binary_generator):
        """Test generating disulfides for selected classes."""
        # Test with valid class IDs
        class_ids = ["+-+++", "-+---"]
        result = binary_generator.generate_for_selected_classes(class_ids, use_class_str=True)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "+-+++" in result
        assert "-+---" in result
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())
        
        # Test with invalid class IDs
        result = binary_generator.generate_for_selected_classes(["invalid"], use_class_str=True)
        assert result == {}
        
        # Test with class column
        result = binary_generator.generate_for_selected_classes(["1", "2"], use_class_str=False)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "1" in result
        assert "2" in result

    def test_generate_for_all_classes(self, binary_generator):
        """Test generating disulfides for all classes."""
        result = binary_generator.generate_for_all_classes()
        assert isinstance(result, dict)
        assert len(result) == 2  # Two classes in the sample data
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())
        
        # Verify all classes are stored in binary_class_disulfides
        assert len(binary_generator.binary_class_disulfides) == 2
        assert all(key in binary_generator.binary_class_disulfides for key in result.keys())

    def test_generate_disulfides_for_class(self, binary_generator):
        """Test the _generate_disulfides_for_class method."""
        # Get the first row from the DataFrame
        row = binary_generator.df.iloc[0]
        
        # Generate disulfides
        disulfide_list = binary_generator._generate_disulfides_for_class(row)
        
        # Verify the result
        assert isinstance(disulfide_list, DisulfideList)
        assert len(disulfide_list) == 243
        assert disulfide_list.pdb_id == f"Class_{row['class']}_{row['class_str']}"
        
        # Check the first disulfide
        first_disulfide = disulfide_list[0]
        assert isinstance(first_disulfide, Disulfide)
        assert first_disulfide.name.startswith(f"{row['class_str']}b_comb")
        
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

    def test_csv_file_not_loaded(self, sample_csv_file):
        """Test error when CSV file is not loaded."""
        generator = DisulfideClassGenerator(csv_file=sample_csv_file)
        generator.df = None
        
        with pytest.raises(ValueError):
            generator.generate_for_class("+-+++")
        
        with pytest.raises(ValueError):
            generator.generate_for_selected_classes(["+-+++"])
        
        with pytest.raises(ValueError):
            generator.generate_for_all_classes()


class TestHelperFunctions:
    """Tests for the helper functions in the DisulfideClassGenerator module."""

    def test_generate_disulfides_for_all_classes(self, sample_csv_file):
        """Test the generate_disulfides_for_all_classes function."""
        result = generate_disulfides_for_all_classes(sample_csv_file, base=2)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_disulfides_for_selected_classes(self, sample_csv_file):
        """Test the generate_disulfides_for_selected_classes function."""
        class_ids = ["1", "2"]
        result = generate_disulfides_for_selected_classes(
            sample_csv_file, class_ids, base=2, use_class_str=False
        )
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(ss_list, DisulfideList) for ss_list in result.values())
        assert all(len(ss_list) == 243 for ss_list in result.values())

    def test_generate_disulfides_for_class_from_csv(self, sample_csv_file):
        """Test the generate_disulfides_for_class_from_csv function."""
        # Test with binary base
        result = generate_disulfides_for_class_from_csv(
            sample_csv_file, class_id="+-+++", base=2
        )
        assert isinstance(result, DisulfideList)
        assert len(result) == 243
        
        # Test with octant base
        result = generate_disulfides_for_class_from_csv(
            sample_csv_file, class_id="1", base=8
        )
        assert isinstance(result, DisulfideList)
        assert len(result) == 243
        
        # Test with invalid base
        with pytest.raises(ValueError):
            generate_disulfides_for_class_from_csv(
                sample_csv_file, class_id="1", base=3
            )


if __name__ == "__main__":
    pytest.main()
