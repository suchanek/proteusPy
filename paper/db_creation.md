# Database Creation

The following steps were performed to create the RCSB disulfide database:

1. Identify disulfide containing proteins in the [RCSB](https://www.rcsb.org). I generated a query using the web-based query tool for all proteins containing one or more disulfide bond. The resulting file consisted of 35,819 IDs.
2. Download the structure files to disk. This resulted in the program [DisulfideDownloader.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideDownloader.py). The download took approximately twelve hours.
3. Extract the disulfides from the downloaded structures. The program [DisulfideExtractor.py](https://github.com/suchanek/proteusPy/blob/master/programs/DisulfideExtractor.py) was used to extract disulfides from the individual structure files. This seemingly simple task was complicated by several factors including:

   1. The PDB file parser contained in Bio.PDB described in [@Hamelyrck_2003] lacked the ability to parse the `SSBOND` records in PDB files. As a result I forked the Biopython repository and updated the `parse_pdb_header.py` file. My fork is available at: [https://github.com/suchanek/biopython]("https://github.com/suchanek/biopython")
   2. Duplicate disulfides contained within a multi-chain protein file.
   3. Physically impossible disulfides, where the $C_\alpha - C_\alpha$ distance is > 8 $\AA$ .
   4. Structures with disordered CYS atoms.

In the end I elected to only use a single example of a given disulfide from a multi-chain entry, and removed any disulfides with a $C_\alpha - C_\alpha$ distance is > 8 $\AA$. This resulted in the current database consisting of 35,808 structures and 120,494 disulfide bonds. To my knowledge this is the only searchable database of disulfide bonds in existence.
