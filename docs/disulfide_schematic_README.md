# Disulfide Schematic Module

## Overview

The `disulfide_schematic` module provides functionality to create publication-ready 2D schematic diagrams of disulfide bonds. These diagrams represent disulfide bonds as graphs where atoms are nodes and bonds are edges, providing a clear visualization of the bond structure.

## Features

- Create 2D schematic diagrams of disulfide bonds
- Support for real disulfide bonds from PDB structures or model disulfides
- Multiple visualization styles: simple, publication, and detailed
- Option to display atom labels and dihedral angles
- Support for various output formats (PNG, SVG, PDF)
- High-resolution output suitable for publication

## Usage

### Basic Usage

```python
import proteusPy as pp
from proteusPy.disulfide_schematic import create_disulfide_schematic

# Load a disulfide from the database
PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)
ss = PDB_SS[0]  # Get the first disulfide

# Create and save a schematic
fig, ax = create_disulfide_schematic(
    disulfide=ss,
    output_file="disulfide_schematic.png",
    show_angles=True,
    style="publication"
)
```

### Creating a Model Disulfide Schematic

```python
from proteusPy.disulfide_schematic import create_disulfide_schematic_from_model

# Create a model disulfide schematic with specific dihedral angles
fig, ax = create_disulfide_schematic_from_model(
    chi1=-60, chi2=-60, chi3=-90, chi4=-60, chi5=-60,
    output_file="model_disulfide_schematic.png",
    show_angles=True
)
```

## Function Reference

### `create_disulfide_schematic`

```python
create_disulfide_schematic(
    disulfide=None,
    output_file=None,
    show_labels=True,
    show_angles=False,
    show_title=True,
    style="publication",
    dpi=300,
    figsize=(8, 6)
)
```

**Parameters:**

- `disulfide` (Disulfide, optional): The disulfide bond object to visualize. If None, creates a model disulfide.
- `output_file` (str, optional): Path to save the output file (supports .svg, .pdf, .png). If None, displays the figure.
- `show_labels` (bool, default=True): Whether to show atom labels.
- `show_angles` (bool, default=False): Whether to show dihedral angles.
- `show_title` (bool, default=True): Whether to show the title with disulfide information.
- `style` (str, default="publication"): Visualization style ("publication", "simple", "detailed").
- `dpi` (int, default=300): Resolution for raster outputs.
- `figsize` (tuple, default=(8, 6)): Figure size in inches.

**Returns:**

- `fig, ax` (tuple): Matplotlib figure and axis objects.

### `create_disulfide_schematic_from_model`

```python
create_disulfide_schematic_from_model(
    chi1=-60, chi2=-60, chi3=-90, chi4=-60, chi5=-60,
    output_file=None,
    show_labels=True,
    show_angles=True,
    style="publication",
    dpi=300,
    figsize=(8, 6)
)
```

**Parameters:**

- `chi1, chi2, chi3, chi4, chi5` (float): Dihedral angles for the disulfide bond.
- `output_file` (str, optional): Path to save the output file (supports .svg, .pdf, .png). If None, displays the figure.
- `show_labels` (bool, default=True): Whether to show atom labels.
- `show_angles` (bool, default=True): Whether to show dihedral angles.
- `style` (str, default="publication"): Visualization style ("publication", "simple", "detailed").
- `dpi` (int, default=300): Resolution for raster outputs.
- `figsize` (tuple, default=(8, 6)): Figure size in inches.

**Returns:**

- `fig, ax` (tuple): Matplotlib figure and axis objects.

## Visualization Styles

- **Simple**: Basic representation with minimal details, suitable for general audience.
- **Publication**: Professional appearance with proper chemical conventions, optimized for journal requirements.
- **Detailed**: Includes additional information like dihedral angles, suitable for technical publications.

## Dependencies

- matplotlib
- networkx
- numpy
- proteusPy

## Example

See the `examples/disulfide_schematic_example.py` script for a complete example of how to use this module.

## Author

Eric G. Suchanek, PhD
