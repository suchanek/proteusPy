import pyrosetta
from pyrosetta import *
from pyrosetta.teaching import *

pyrosetta.init()


init()
# Set up score function including disulfide torsion energy
sfxn = get_fa_scorefxn()
sfxn.set_weight(rosetta.core.scoring.dslf_fa13, 1.0)

pose = pyrosetta.toolbox.pose_from_rcsb("2q7q")
res75 = pose.pdb_info().pdb2pose("D", 75)
res140 = pose.pdb_info().pdb2pose("D", 140)

emap = EMapVector()
sfxn.eval_ci_2b(pose.residue(res75), pose.residue(res140), pose, emap)
print("FA Atr:", emap[fa_atr])
print("FA Rep:", emap[fa_rep])
print("FA Sol:", emap[fa_sol])

### END SOLUTION
