# Cluster Analysis of Disulfide Bonds in Proteins of Known Structure
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/19/23 -egs-
# Cα Cβ Sγ

import math

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly_express as px
import pyvista as pv
import seaborn as sns
from pyvista import set_plot_theme

import proteusPy
from proteusPy import *
from proteusPy.DisulfideBase import *
from proteusPy.ProteusGlobals import *

# print(pv.Report())

plt.style.use("dark_background")

# ipyvtklink
# pv.set_jupyter_backend('ipyvtklink')

set_plot_theme("document")

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = "/Users/egs/PDB/"

# location of cleaned PDB files
PDB = "/Users/egs/PDB/good/"

# location of the compressed Disulfide .pkl files
MODELS = f"{PDB_BASE}data/"

# default parameters will read from the package itself.
PDB_SS = DisulfideLoader(verbose=True, subset=True)


# retrieve the torsions dataframe
from proteusPy.proteusGlobals Torsion_DF_Cols

_SSdf = PDB_SS.getTorsions()

# there are a few structures with bad SSBonds. Their
# CA distances are > 7.0. We remove them from consideration
# below

_near = _SSdf["ca_distance"] < 9.0

# entire database with near cutoff of 9.0
SS_df = _SSdf[_near]
SS_df = SS_df[Torsion_DF_Cols].copy()


from sklearn.mixture import GaussianMixture

n_clusters = 8

_cols = [
    "chi1",
    "chi2",
    "chi3",
    "chi4",
    "chi5",
    "torsion_length",
    "energy",
    "ca_distance",
]

tor_df = SS_df[_cols]
tor_df.head(1)

"""
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(tor_df)
cluster_labels = gmm_model.predict(tor_df)

X = pd.DataFrame(tor_df)
X['cluster'] = cluster_labels

for k in range(n_clusters):
    data = X[X['cluster'] == k]
    plt.scatter(data['torsion_length'], data['ca_distance'], s=2)

plt.show()
"""

import seaborn as sns

# takes over an hour for full dataset
from sklearn.cluster import AffinityPropagation

n_clusters = 6

_cols = ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
tor_df = SS_df[_cols].copy()

import seaborn as sns

# takes over an hour for full dataset
from sklearn.cluster import AffinityPropagation

n_clusters = 3

_cols = ["chi1", "chi2", "chi3", "chi4", "chi5", "torsion_length"]
tor_df = SS_df[_cols].copy()

X = tor_df.copy()

aff_model = AffinityPropagation(max_iter=100, random_state=25)
# takes 51 min with full dataset
X["cluster"] = aff_model.fit_predict(X[["torsion_length"]])

fig, ax = plt.subplots()
ax.set(title="Affinity Propagation")
sns.scatterplot(
    x="chi1", y="torsion_length", data=X, hue="cluster", ax=ax, size=2
)
