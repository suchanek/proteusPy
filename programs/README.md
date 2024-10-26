# proteusPy Programs

This directory contains programs used by the ``proteusPy`` package, written by Eric G. Suchanek, PhD. I list the most relevant and their descriptions in this file. The programs listed below are up and running, and are in 'production'. Some of the other programs are half-baked and not used in production.

* `DisulfideExtractor_mp.py` - this is the main program used to extract Disulfide Bonds from the RCSB files. The program uses multi-processing and can parse over 36,000 files on my MacbookPro M3 Max in a little over 2 minutes!
* `DisulfideExtractor_threaded.py` - this is the multi-threaded version of the Extractor, and is the version prior to the multiprocessing version. Historical, but shows how to do multi-threading.
* `DisulfideClass_Analysis.py` - this program uses the existing database to create consensus disulfide structures by either binary or sextant class distribution. It can also generate graphs showing the statistics for each class.
* `DisulfidePruner.py` - this program removes duplicat disulfides from the database. In general this is no longer needed, since it's embodied in the extractor.
* `rcsb_download.py` - this program can download individual RCSB files.
* f`ilter_pdb.py` - this program attempts to parse SSBOND records using the new ssparser.py functions.
* `DBViewer.py` - this is the ``panel``-based viewer for the RCSB database. Launch with ``panel serve full_path/DBViewer.py --auto-reload --show &``
