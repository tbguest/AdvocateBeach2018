#!/usr/bin/env python3
"""
Created on Thu Nov 22 15:47:31 2018

@author: Owner
"""

# %reset

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import stats
from scipy.signal import correlate
import json
from scipy.stats.stats import pearsonr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.colors as mcolors
import matplotlib.colors
import glob

# %matplotlib qt5

plt.close('all')

# change default font size
plt.rcParams.update({'font.size': 10})


def save_figures(dn, fn, fig):
    ''' Saves png and pdf of figure.

    INPUTS
    dn: save directory. will be created if doesn't exist
    fn: file name WITHOUT extension
    fig: figure handle
    '''

    if not os.path.exists(dn):
        try:
            os.makedirs(dn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig.savefig(os.path.join(dn, fn + '.png'), dpi=1000, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.pdf'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.eps'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.jpg'), dpi=1000, transparent=True)

homechar = os.path.expanduser("~") # linux


        # output = {'correlation':r1_all, 'y':y1[0]}
        # # save new variables
        # fout = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
        #                  "processed","survey_data", "dz_dmgs_correlations")
        #
        # np.save(os.path.join(fout, fn), output)


dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                 "processed","survey_data", "dz_dmgs_correlations")

r = []
p = []
lo = []
hi = []
y = []

for file in glob.glob(os.path.join(dn, "*.npy")):

    jnk = np.load(file, allow_pickle=True).item()

    r.append(jnk['correlation'][0])
    p.append(jnk['correlation'][1])
    lo.append(jnk['correlation'][2])
    hi.append(jnk['correlation'][3])

    if 'longshore2' not in file:
        y.append(jnk['y'])
    else:
        y.append(jnk['y'] + 0.25)



fig, ax = plt.subplots(1,1,figsize=(3,4), num='corrs')
ax.plot(r, y, 'k.')
ax.plot([lo, hi], [y, y], 'k-')
yl = ax.get_ylim()
xl = ax.get_xlim()
ax.plot([0,0], yl, 'k--')
ax.axhspan(-9, yl[0], alpha=0.25, color='grey')
ax.invert_yaxis()
ax.set_xlabel('r')
ax.set_ylabel('y [m]')
ax.autoscale(enable=True, axis='y', tight=True)
fig.tight_layout()



saveFlag = 1
# export figs
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD')

    save_figures(savedn, 'dz_dmgs_correlations_vs_y', fig)
