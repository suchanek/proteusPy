# Analysis of Disulfide Bonds in Proteins of Known Structure 
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/19/23 -egs-
# Cα Cβ Sγ

import math
import matplotlib
import matplotlib.pyplot as plt

import plotly_express as px
#import seaborn as sns

import proteusPy
from proteusPy import *
from proteusPy.Disulfide import *

from proteusPy.ProteusGlobals import *

import pandas as pd
import pyvista as pv
from pyvista import set_plot_theme
import time


plt.style.use('dark_background')

# ipyvtklink
#pv.set_jupyter_backend('ipyvtklink')

set_plot_theme('document')

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files
PDB = '/Users/egs/PDB/good/'

# location of the compressed Disulfide .pkl files
MODELS = f'{PDB_BASE}models/'

# when running from the repo the local copy of the Disulfides is in proteusPy/data
# the locations below represent the actual location on the dev drive.
# location for PDB repository
# takes 
PDB_BASE = '/Users/egs/PDB/'

# location of the compressed Disulfide .pkl files. Currently I don't have the entire
# dataset in the repo, so to load the full dataset I point to my dev drive

DATA = f'{PDB_BASE}data/'

PDB_SS = DisulfideLoader(verbose=True, subset=True, datadir=DATA)


# Set the figure sizes and axis limits.
DPI = 220
WIDTH = 6.0
HEIGHT = 3.0
TORMIN = -179.0
TORMAX = 180.0
GRIDSIZE = 20


#
#
# retrieve the torsions dataframe
from proteusPy.Disulfide import Torsion_DF_Cols

_SSdf = PDB_SS.getTorsions()
# there are a few structures with bad SSBonds. Their
# CA distances are > 7.0. We remove them from consideration
# below

_far = _SSdf['ca_distance'] >= 9.0
_near = _SSdf['ca_distance'] < 9.0

SS_df_Far = _SSdf[_far]

# entire database
SS_df = _SSdf[_near]

SS_df = SS_df[Torsion_DF_Cols].copy()
SS_df.describe()

# The distances are held in the overall Torsions array. We get this and sort
distances = PDB_SS.getTorsions()
distances.sort_values(by=['ca_distance'], ascending=False, inplace=True)

distances.head(20)

# 1
from sklearn.mixture import GaussianMixture
n_clusters = 4

_cols = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'torsion_length', 'energy', 'ca_distance']

tor_df = SS_df[_cols]
tor_df.head(1)
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(tor_df)
cluster_labels = gmm_model.predict(tor_df)
X = pd.DataFrame(tor_df)
X['cluster'] = cluster_labels
for k in range(n_clusters):
    data = X[X['cluster'] == k]
    plt.scatter(data['torsion_length'], data['ca_distance'], s=2)

#plt.show()

# 2
# takes over an hour for full dataset
from sklearn.cluster import SpectralClustering
import seaborn as sns

_cols = ['chi3', 'torsion_length', 'energy']
#_cols = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'torsion_length']
#cols = ['ca_distance', 'chi3', 'energy', 'torsion_length']

# tor_df = SS_df[['chi1', 'chi2', 'chi3', 'chi4', 'chi5']].copy()

tor_df = SS_df[_cols].copy()

X = tor_df
n_clusters = 4

scm_model = SpectralClustering(n_clusters=n_clusters, random_state=25,
                                n_neighbors=8, affinity='nearest_neighbors')
# takes 51 min with full dataset
'''
print(f'Spectral Clustering starts')
X['cluster'] = scm_model.fit_predict(X[['torsion_length']])

fig, ax = plt.subplots()

ax.set(title='Spectral Clustering')
sns.scatterplot(x='chi3', y='torsion_length', data=X, hue='cluster', ax=ax, size=2)
'''

#
# takes over an hour for full dataset
from sklearn.cluster import AffinityPropagation

start = time.time()

n_clusters = 6
_cols = ['ca_distance', 'torsion_length', 'energy']

tor_df = SS_df[_cols].copy()

X = tor_df.copy()

aff_model = AffinityPropagation(max_iter=200, random_state=25)
# takes 51 min with full dataset
X['cluster'] = aff_model.fit_predict(X[['torsion_length']])


fig, ax = plt.subplots()
ax.set(title='Affinity Propagation')
sns.scatterplot(x='torsion_length', y='energy', data=X, hue='cluster', ax=ax, size=2)

from sklearn.mixture import GaussianMixture
n_clusters = 4

_cols = ['torsion_length', 'energy', 'ca_distance']

tor_df = SS_df[_cols]
tor_df.head(1)
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(tor_df)
cluster_labels = gmm_model.predict(tor_df)
X = pd.DataFrame(tor_df)
X['cluster'] = cluster_labels
for k in range(n_clusters):
    data = X[X['cluster'] == k]
    plt.scatter(data['torsion_length'], data['ca_distance'], s=2)

plt.show()


end = time.time()
elapsed = end - start

print(f'Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')




