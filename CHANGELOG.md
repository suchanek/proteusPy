# proteusPy ChangeLog

Notable changes to the ``proteusPy`` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
