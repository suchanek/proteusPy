<img src="logo.png" alt="ProteusPy Logo" style="width:25%;">

# ChangeLog

Notable changes to the ``proteusPy`` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.99.35] - unreleased

### Added

- ``DisulfideEnergy`` class to decompose the disulfide torsional strain calculations using the standard calculation and also the equation used by Hogg et al. in their Allosteric Disulfide Bond paper. @TODO refactor ``Disulfide`` to use this class rather than the currently built-in energy functions.
- Moved several notebooks into the **examples/** directory.
-

### Changed

- Added ``rich`` text formatting to the logging functions, yielding more attractive log messages.
- Cleaned up the global logging/file handling in ``logger_config.py``.
- Enhanced the ``DisulfideLoader.summary()`` function for readability.

### Fixed

- Low level bug with global logger which caused duplicate log messages.
- Turned on translation to center of mass for visualization to center each disulfide.

### Issues

- ``Disulfide.spin()`` only works under MacOS. Throws a plotter error under Windows.
-

## [v0.99.34] - 2025-04-02

### Added

- statistical calculations for classes in DisulfideClass_Analysis.py, writes the .pkl to the appropriate dir. The .csv metrics file is saved under $PDB/data.
- ``DisulfideClassGenerator.py``. This new class manages all aspects of the generation
  of disulfide structures that represent the overall envelope of structures accessible
  to a given structural class consensus disulfide. For instance, if a binary class
  (00000b) has a consensus structure of chi1-chi5 +/- five degrees it can generate the
  243, (3^5) structures and calculate statistics for this 'tree'.
- ``display_class_disufides`` entrypoint added. This utilizes the new class to display a given
  disulfide class in a separate window using the ``DisulfideList.display_overlay()`` function.
- ``hexbin_plot`` entrypoint added. This program creates 3D interactive plots showing dihedral angle correlations between left-handed and right-handed disulfides.
- ``proteusPy.DisulfideBase.Disulfide.TorsionEnergyKJ`` property and calculation to use Hogg's DSE potential function.
- Enhanced ``DisulfideLoader`` to accept a percentile cutoff. This then calculates proper Ca and Sg distance cutoffs and applies them to the master disulfide list upon instantiation. This obviates the need for explicit Ca and Sg cutoffs.

### Changed

- ``DisulfideClass_Analysis.py`` now creates the binary and octant torsion metrics
  files needed for the new ``DisulfideClassGenerator`` class. These are needed upon
  class instantiation and are bundled into the package. Note that the octant class
  metrics file will be dependent on the overall cutoff used during the program run.
  I typically use 0.04, which generates 329 overall consensus structures.
- Removed explicit ca and sg cutoffs from ``DisulfideLoader.Load_PDB_SS()``. Now
  it uses percentile only.

### Fixed

- class string calculation was flipped.
- ``DisulfideLoader`` cutoff values weren't propagating properly
- continued catching small bugs

## [v0.99.33] - 2025-03-05

### Fixed

- a few small issues with testing under Linux
- Makefile tweaks

## [v0.99.32] - 2025-03-05

### Added

- disulfide_schematic.py which has two functions for creating schematic
drawings of Disulfides in various styles:
  - ``create_disulfide_schematics()``
  - ``create_disulfide_schematic_from_model()``
  - test function in the tests/ directory
  - script endpoint: proteusPy.create_disulfide_schematics

### Fixed

- Finished refactoring for ``DisulfideIO.py``
- Corrected parameter order in ``DisulfideBase.DisulfideList.display_overlay()``
- Corrected the ``.screenshot()`` function to work correctly on Windows.
- Corrected Makefile installation target to install from the wheel, not repo

## [v0.99.31] - 2025-2-23

### Added

- ``DisulfideList.Average_Sg_Distance`` property
- Disulfide List info added to the ``rcsb_viewer.py`` info pane

### Fixed

- ``rcsb_viewer.py`` - refactored for new ``DisulfideVisualization`` classes.
- Pushed to both DockerHub and GitHub repositories.
-

## [v0.99.3] - 2024-2-22

### Added

- Stronger type checking across many functions
- Additional error checking for slicing lists
- Several new Unittests

### Changed

- Refactored both ``Disufulfide`` and ``DisulfideList`` and moved plotting and statistics into new classes ``DisulfideVisualization`` and ``DisulfideStats``.
- Simplification and unification of accessing Disulfides via class strings
- Generalized Disulfide class creation, can create up to 26-fold classes now.

### Fixed

- DisulfideList.AverageConformation now uses a circular mean function to correctly handle the nature of -180 - 180 degree dihedral angle averaging.

## [v0.99.1.dev0] - 2024-2-07 unreleased

### Added

- Disulfide.spin() - Spin the disulfide about the Y axis.
- Test_DisplaySS.py - Unittest improvements.

### Changed

- Completely rewrote ``DisulfideClass_Constructor`` methods that generate the binary and octant class strings for simplicity.
- The new ``DisulfideClass_Constructor.get_segment()`` is a more general version of ``DisulfideClass_Constructor.get_eighth_quadrant()``.

## [v0.98.5] - 2024-1-23

### Added

- static method ``DisufideClass_Constructor.class_string_from_dihedral()`` - return the binary or octant class string for a single or array of dihedrals.
- properties ``Disulfide.binary_class_string`` and ``Disulfide.octant_class_string``
- ``DisulfideLoader.sslist_from_class()`` - now uses the index value for a specific disulfide that matches the classID from the TorsionDF dataframe. This allows for direct access to the DisulfideList containing the Disulfides via ``loader[index]``, which is significantly faster than looking up via ``loader[disulfide_name]``.

### Changed

- vectorized binary and octant class string construction
- added binary and octant class strings to the master ``loader.TorsionDF`` DataFrame.
- Rewrote ``programs/DisulfideClass_Extractor.py`` to use the index-based address scheme described above. This resulted in about a 20x speedup in the program!

## [v0.98.4] - 2024-1-17

### Added

- ``DisulfideList.plot_distances()``
- ``DisulfideList.plot_deviation_histograms()``
- Added the ability to access disulfide class strings directly through ``DisulfideLoader`` with indexing.

### Changed

- ``Load_PDB_SS(verbose=True)`` now calls ``loader.describe()`` to print statistics for the database
- Optimized ``DisulfideList.create_deviation_dataframe()``.
- Moved various class plotting routines into ``DisulfideLoader`` class.
- Cleaned up ``DisulfideClasses.py``

## [v0.98.3] - 2024-1-12

### Added

- ``DisulfideExtractor_mp.py`` moved into the package as a callable module.

  ``proteusPy.DisulfideExtractor`` from command line will launch the program

- Incorporated consensus structures into the ``DisulfideClass_Constructor`` object. This presumes these have been generated. The consensus structures are created through the program ``DisulfideClass_Analysis.py``.

### Changed

- Corrected an error in ``DisulfideLoader`` that failed to initialize the torsion dataframe properly after filtering.
- Change to setup.py - 2q7q_seqsim.csv was not being included

### Fixed

- Sg_distance was not being calculated with ``Disulfide.build_yourself()``
- phi and psi were not correctly populating in the torsion dataframe.
- There was a subtle error in the ``DisulfideLoader`` initialization that led to internal database inconsistencies after filtering. This has been corrected.

## [v0.98.2] - 2024-12-30

### Added

- ``qt5viewer.py`` moved into the package as a callable module.

``proteusPy.qt5viewer`` from command line will launch the program

### Changed

- logging cleanup in ``DisulfideLoader.py`` and ``DisulfideList.py``.
- ongoing documentation tweaks, cleanup

## [v0.98.1] - 2024-12-30

### Added

- qt5viewer.py moved into the package as a callable module

``proteusPy.qt5viewer`` from command line will launch the program

### Changed

- moved to Python 3.12

## [v0.98] - 2024-12-24

### Added

- Dynamic resolution for the rcsb_viewer List view.

### Changed

- ``DisulfideList`` code optimization

## [V0.97.17] - 2024-11-30

### Added

- Additional work on rcsb_viewer.py
- Automation scripts for Docker builds

## [V0.97.16] - 2024-11-22

### Added

- One can now access a disulfide by name directly from the loader with:

  ```
    pdb = Load_PDB_SS()
    ss = pdb["2q7q_75D_140D"]
  ```

  In prior versions one would need to use the loader.get_by_name() function.

### Removed

- Removed the ``programs/rcsb_viewer.py`` program. The viewer now lives only in the ``viewer`` directory and can be invoked directly from the command line with:

```console
$panel serve ~/repos/proteusPy/viewer/rcsb_viewer.py --show
```

## [V0.97.15] - 2024-11-10

### Added

- Unified the disulfide viewers such that the rcsb_viewer.py program will work either stand-alone or in Docker.
- Added workflows to build the Docker images on GitHub and Docker Hub
- Made pyqt5 an optional install, pip install proteusPy[pyqt5] adds it back.

## [V0.97.11]

### Added

- Renderers:
  - Docker image for the pyVista renderer. This lives under proteusPy/viewer and is deployed on Docker hub under ``egsuchanek/rcsb_viewer``.
Launch with: ``docker -d -p 5006:5006 egsuchanek/rcsb_viewer:latest. Works under MacOSX and Linux.
  - The standalone Panel-based version lives in ``proteusPy/programs/DBViewer.py``. Launch with: ``panel serve path_to_DBViewer.py --autoreload &``
  - PyQt5 version. This lives in ``proteusPy/programs/QT5Viewer.py``. It is the most advanced version, but I'm unable to build under Linux. My intent was to deploy this via Docker, but can't get PyQt5 to build currently.

- ``DisulfideList.center_of_mass`` - returns the ``Vector3D`` center of mass for the list.
- ``DisulfideList.translate()`` - adds the input ``Vector3D`` object from the list, destructively modifying the coordinates. Used primarily in rendering functions to center the list at ``Vector3D([0,0,0])``.
- ``Disulfide.translate()`` - translates the input Disulfide by the input ``Vector3D``.

### Issues

- I cannot get the QT5 viewer to build under linux. The pyQt5 library won't install.

## [V0.97.10]

### Added

- Disulfide QT5 viewer development, improvement
- ``DisulfideList.display()`` added to provide a summary of the input DisulfideList
- Additional analytics

### Fixed

- Analysis of the PDB entry files revealed yet another parsing issue. There are many structures that contain disulfides referring to themselves, ie 25A-25A. These had not been caught in any prior release and were revealed while I was working on the filtering code.

## [v0.97.9] - 2024-10-07

### Added

Created a Disulfide viewer using pyQt5 library. This program is under programs/viewer.py. It's basic, but provides an easy way to visualize disulfides within the database. The single checkbox toggles between individual disulfide, or overlaid. The latter shows all of the disulfides for the given protein entry, overlaid onto a common coordinate system. I continue to tweak the code, but the core is stable.

## [v0.97.8] - 2024-09-16

### Added

There have been many internal changes to the package since the official release. I list the most relevant below:

- Completely re-wrote the PDB parser, removing the dependency on my Biopython fork. This has led to great improvements in the overall Disulfide extraction process.
- Implemented multi-processing for the Disulfide_Extractor program. It's possible to extract disulfides from over 37,000 PDB files in under 3 minutes on my 2024 M3 Max MacBook Pro, using 14 cores. This process initially took over 1.5 hours!
- Implemented Octant (8-fold) class construction.
- Implemented bond angle and bond distance ideality calculations in order to intelligently extract high-quality disulfides from the database.
- Implemented dynamic DisulfideLoader creation from the master SS list extracted by the DisulfideExtractor.py program.

### Deprecated

- All Biopython references will be ultimately removed.

## [v0.96.31] - 2024-08-06

### Added

- Bump release of ProteusPy with core functionalities corresponding to JOSS paper.
- Publication of JOSS paper

## [v0.96.3] - 2024-07-18

### Added

- Initial release of ProteusPy with core functionalities.
