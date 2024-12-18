# Changelog for proteusPy

All notable changes to the ``proteusPy`` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.98] - 2024-12-14

### Added

- Dynamic resolution for the rcsb_viewer List view.
- ``DisulfideList`` code optimization


## [v0.97.10] - 2024-10-14

### Added

- DisulfideList.describe() - prints a summary of the input ``DisulfideList``
- Improvement in internal logging. The function ``configure_master_logger`` sets up a file logger that can be used to collect info, warning and error output. This is particularly useful when running the ``DisulfideExtractor_mp.py``

### Changed

- Disulfide viewer development continues
- Continued code optimizations

### Deprecated

- Biopython references will be ultimately removed.

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

## [v0.96.31] - 2024-08-06

### Added

- Bump release of ProteusPy with core functionalities corresponding to JOSS paper.
- Publication of JOSS paper

## [v0.96.3] - 2024-07-18

### Added

- Initial release of ProteusPy with core functionalities.
