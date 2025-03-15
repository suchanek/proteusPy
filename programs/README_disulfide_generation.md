# Disulfide Generation and Analysis Tools

This collection of scripts provides tools for generating and analyzing disulfide conformations for different structural classes based on the data in the `binary_class_metrics_0.00.csv` file.

## Overview

The scripts in this collection allow you to:

1. Generate disulfide conformations for different structural classes
2. Analyze the energy distribution of the generated disulfides
3. Visualize the disulfides and their properties
4. Save the generated disulfides for further analysis

## Files

- `generate_class_disulfides.py`: Core module containing the DisulfideClassGenerator class and functions for generating disulfides
- `example_generate_disulfides.py`: Simple example demonstrating how to use the generation functions
- `example_class_generator.py`: Example demonstrating how to use the DisulfideClassGenerator class
- `analyze_class_disulfides.py`: Advanced script for analyzing and visualizing generated disulfides
- `compare_class_disulfides.py`: Script for comparing disulfides from different structural classes

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- proteusPy package

## Usage

### Generating Disulfides for a Specific Class

```python
from generate_class_disulfides import generate_disulfides_for_class_from_csv

# Path to the CSV file
csv_file = "binary_class_metrics_0.00.csv"

# Generate disulfides for a specific class
class_id = "22222"  # The "+++++" class
disulfide_list = generate_disulfides_for_class_from_csv(csv_file, class_id)

# Print information about the generated disulfides
print(f"Generated {len(disulfide_list)} disulfides for class {class_id}.")
print(f"Average energy: {disulfide_list.average_energy:.2f} kcal/mol")
```

### Generating Disulfides for Multiple Classes

```python
from generate_class_disulfides import generate_disulfides_for_selected_classes

# Path to the CSV file
csv_file = "binary_class_metrics_0.00.csv"

# Generate disulfides for multiple classes
selected_classes = ["22222", "00000", "02222"]
class_disulfides = generate_disulfides_for_selected_classes(csv_file, selected_classes)

# Print information about the generated disulfides
for class_id, disulfide_list in class_disulfides.items():
    print(f"Class {class_id}: Generated {len(disulfide_list)} disulfides.")
    print(f"Average energy: {disulfide_list.average_energy:.2f} kcal/mol")
```

### Analyzing Disulfides

```python
from analyze_class_disulfides import analyze_energy_distribution, analyze_dihedral_distributions

# Analyze the energy distribution
analyze_energy_distribution(disulfide_list, class_id, class_str)

# Analyze the dihedral angle distributions
analyze_dihedral_distributions(disulfide_list, class_id, class_str)
```

### Comparing Disulfides from Different Classes

```python
from compare_class_disulfides import (
    compare_energy_distributions,
    compare_dihedral_distributions,
    compare_minimum_energy_disulfides,
    compare_average_conformations
)

# Generate disulfides for multiple classes
selected_classes = ["22222", "00000", "00200", "02020", "20220"]
class_disulfides = generate_disulfides_for_selected_classes(csv_file, selected_classes)

# Get class names from the CSV file
df = pd.read_csv(csv_file)
class_names = {class_id: df[df['class'] == class_id].iloc[0]['class_str'] for class_id in selected_classes}

# Compare energy distributions
compare_energy_distributions(class_disulfides, class_names)

# Compare dihedral angle distributions
compare_dihedral_distributions(class_disulfides, class_names)

# Compare minimum energy disulfides
min_energy_disulfides = compare_minimum_energy_disulfides(class_disulfides, class_names)

# Compare average conformations
avg_conformation_disulfides = compare_average_conformations(class_disulfides, class_names)
```

### Using the DisulfideClassGenerator Class

The `DisulfideClassGenerator` class provides an object-oriented approach to generating disulfides:

```python
from proteusPy.generate_class_disulfides import DisulfideClassGenerator

# Create a generator instance and load the CSV file
generator = DisulfideClassGenerator("binary_class_metrics_0.00.csv")

# Generate disulfides for a specific class using class_str
disulfide_list = generator.generate_for_class("+++++", use_class_str=True)

# Generate disulfides for a specific class using class ID
disulfide_list = generator.generate_for_class("22222", use_class_str=False)

# Generate disulfides for multiple classes
selected_classes = ["+++++", "-----", "-+---"]  # RH Spiral, LH Spiral, and RH Staple
class_disulfides = generator.generate_for_selected_classes(selected_classes, use_class_str=True)

# Generate disulfides for all classes
all_class_disulfides = generator.generate_for_all_classes()

# Find the minimum energy disulfide for each class
for class_id, disulfide_list in class_disulfides.items():
    min_energy_disulfide = min(disulfide_list, key=lambda ss: ss.energy)
    print(f"Class {class_id}: {min_energy_disulfide.energy:.2f} kcal/mol")
```

See `example_class_generator.py` for a complete example of using the DisulfideClassGenerator class.

## How It Works

For each structural class in the CSV file, the scripts extract the mean and standard deviation values for each of the 5 dihedral angles (chi1 through chi5). For each dihedral angle, three values are considered:

1. mean - std
2. mean
3. mean + std

With 5 dihedral angles and 3 possible values for each, this results in 3^5 = 243 combinations per class. The scripts generate a disulfide conformation for each combination, resulting in a comprehensive exploration of the conformational space for each structural class.

## Example Workflow

1. Generate disulfides for a specific class:

   ```
   python example_generate_disulfides.py
   ```

2. Analyze the generated disulfides:

   ```
   python analyze_class_disulfides.py
   ```

3. Compare disulfides from different classes:

   ```
   python compare_class_disulfides.py
   ```

## Output Files

The scripts generate several output files:

- `class_{class_id}_disulfides.pkl`: Pickle file containing all generated disulfides for a class
- `class_{class_id}_min_energy_disulfide.pkl`: Pickle file containing the disulfide with the minimum energy
- `class_{class_id}_avg_conformation_disulfide.pkl`: Pickle file containing a disulfide with the average conformation
- `class_{class_id}_energy_distribution.png`: Plot of the energy distribution
- `class_{class_id}_dihedral_distributions.png`: Plot of the dihedral angle distributions

When comparing classes, additional files are generated:

- `class_energy_comparison.png`: Plot comparing energy distributions between classes
- `class_dihedral_comparison.png`: Plot comparing dihedral angle distributions between classes
- `class_min_energy_comparison.png`: Bar chart comparing minimum energies between classes
- `class_avg_conformation_comparison.png`: Radar chart comparing average conformations between classes

## Notes

- The generated disulfides can be visualized using the `display()` method of the `Disulfide` class, but this requires a graphical environment.
- The scripts are designed to work with the `binary_class_metrics_0.00.csv` file, which contains the mean and standard deviation values for each dihedral angle for different structural classes.
- The energy of each disulfide is calculated using the formula implemented in the `Disulfide` class.

## References

- Schmidt, B., Ho, L., & Hogg, P. J. (2006). Allosteric disulfide bonds. *Biochemistry*, 45(24), 7429-7433.
- Suchanek, E. G. (2025). proteusPy: A Python package for the analysis and modeling of protein structures with an emphasis on disulfide bonds.
