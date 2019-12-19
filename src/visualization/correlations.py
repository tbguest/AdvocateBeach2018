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
plt.rcParams.update({'font.size': 12})


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

# dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
#                  "processed","survey_data", "dz_dmgs_correlations")

dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                 "processed","survey_data", "reprocessed_x10", "dz_dmgs_correlations")

r = []
p = []
lo = []
hi = []
y = []

r_highenergy = []
p_highenergy = []
lo_highenergy = []
hi_highenergy = []
y_highenergy = []

r_lowenergy = []
p_lowenergy = []
lo_lowenergy = []
hi_lowenergy = []
y_lowenergy = []

r_1 = []
p_1 = []
lo_1 = []
hi_1 = []
y_1 = []

r_2 = []
p_2 = []
lo_2 = []
hi_2 = []
y_2 = []

for file in glob.glob(os.path.join(dn, "*.npy")):

    if 'cross_shore' in file:
        continue

    jnk = np.load(file, allow_pickle=True).item()

    r.append(jnk['correlation'][0])
    p.append(jnk['correlation'][1])
    lo.append(jnk['correlation'][2])
    hi.append(jnk['correlation'][3])

    r_highenergy.append(jnk['correlation_highenergy'][0])
    p_highenergy.append(jnk['correlation_highenergy'][1])
    lo_highenergy.append(jnk['correlation_highenergy'][2])
    hi_highenergy.append(jnk['correlation_highenergy'][3])

    r_lowenergy.append(jnk['correlation_lowenergy'][0])
    p_lowenergy.append(jnk['correlation_lowenergy'][1])
    lo_lowenergy.append(jnk['correlation_lowenergy'][2])
    hi_lowenergy.append(jnk['correlation_lowenergy'][3])

    # for highlighting LT1 and LT2 in plot
    if 'longshore1' in file:
        r_1.append(jnk['correlation'][0])
        p_1.append(jnk['correlation'][1])
        lo_1.append(jnk['correlation'][2])
        hi_1.append(jnk['correlation'][3])
        y_1.append(jnk['y'])

    if 'longshore2' in file:
        r_2.append(jnk['correlation'][0])
        p_2.append(jnk['correlation'][1])
        lo_2.append(jnk['correlation'][2])
        hi_2.append(jnk['correlation'][3])
        y_2.append(jnk['y'] + 0.25)


    if 'longshore2' not in file:
        y.append(jnk['y'])
        y_highenergy.append(jnk['y'])
        y_lowenergy.append(jnk['y'])
    else:
        y.append(jnk['y'] + 0.25)
        y_highenergy.append(jnk['y'] + 0.25)
        y_lowenergy.append(jnk['y'] + 0.25)



fig, ax = plt.subplots(1,1,figsize=(3,4), num='corrs')
ax.plot(r, y, 'k.')
ax.plot([lo, hi], [y, y], 'k-')
ax.plot(r_1, y_1, 'C1.')
ax.plot([lo_1, hi_1], [y_1, y_1], 'C1-')
ax.plot(r_2, y_2, 'C0.')
ax.plot([lo_2, hi_2], [y_2, y_2], 'C0-')
yl = ax.get_ylim()
xl = ax.get_xlim()
ax.plot([0,0], yl, 'k--')
ax.plot(xl, [-14.14, -14.14], 'k-', linewidth=0.5)
# ax.axhspan(-9, yl[0], alpha=0.25, color='grey')
ax.invert_yaxis()
ax.set_xlabel('r')
ax.set_ylabel('y [m]')
ax.autoscale(enable=True, axis='y', tight=True)
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()

fig2, ax2 = plt.subplots(1,1,figsize=(3,4), num='corrs2')
ax2.plot(r_highenergy, y_highenergy, 'r.')
ax2.plot([lo_highenergy, hi_highenergy], [y_highenergy, y_highenergy], 'r-')
ax2.plot(r_lowenergy, y_lowenergy, 'k.')
ax2.plot([lo_lowenergy, hi_lowenergy], [y_lowenergy, y_lowenergy], 'k-')
# ax.plot(r_1, y_1, 'k.')
# ax.plot([lo_1, hi_1], [y_1, y_1], 'k-')
# ax.plot(r_2, y_2, 'k.')
# ax.plot([lo_2, hi_2], [y_2, y_2], 'k-')
yl = ax2.get_ylim()
xl = ax2.get_xlim()
ax2.plot([0,0], yl, 'k--')
ax2.axhspan(-9, yl[0], alpha=0.25, color='grey')
ax2.invert_yaxis()
ax2.set_xlabel('r')
ax2.set_ylabel('y [m]')
ax2.autoscale(enable=True, axis='y', tight=True)
fig2.tight_layout()

(9+13*5+15*6+17*2)/14

saveFlag = 0
# export figs
if saveFlag == 1:
    # savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD')
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD','reprocessed')

    save_figures(savedn, 'dz_dmgs_correlations_vs_y', fig)
