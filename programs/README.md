# proteusPy Programs

This directory contains programs used by the ``proteusPy`` package, written by Eric G. Suchanek, PhD. I list the most relevant and their descriptions in this file. The programs listed below are up and running, and are in 'production'. Some of the other programs are half-baked and not used in production.

* `DisulfideClass_Analysis.py` - this program uses the existing database to create consensus disulfide structures by either binary or sextant class distribution. It can also generate graphs showing the statistics for each class.
* `DisulfidePruner.py` - this program removes duplicat disulfides from the database. In general this is no longer needed, since it's embodied in the extractor.
* `rcsb_download.py` - this program can download individual RCSB files.
* `filter_pdb.py` - this program attempts to parse SSBOND records using the new ssparser.py functions.
