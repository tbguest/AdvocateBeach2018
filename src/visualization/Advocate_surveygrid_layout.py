
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 16})


params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)


%matplotlib qt5


def save_figures(dn, fn, fig):
    ''' Saves png and pdf of figure.

    INPUTS
    dn: save directory. will be created if doesn't exist
    fn: file name WITHOUT extension
    fig: figure handle
    '''

    # dn0 = os.path.join(dn, 'png')
    # dn1 = os.path.join(dn, 'pdf')
    # dn2 = os.path.join(dn, 'eps')
    # dn3 = os.path.join(dn, 'jpg')

    if not os.path.exists(dn):
        os.makedirs(dn)
    # if not os.path.exists(dn1):
    #     os.makedirs(dn1)
    # if not os.path.exists(dn2):
    #     os.makedirs(dn2)
    # if not os.path.exists(dn3):
    #     os.makedirs(dn3)

    fig.savefig(os.path.join(dn, fn + '.png'), dpi=1000, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.pdf'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.eps'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.jpg'), dpi=1000, transparent=True)



CT1_y = np.linspace(-30, 60, 31)
CT1_x = np.zeros(len(CT1_y))

LT1_x = np.linspace(-50, 25, 26)
LT1_y = -5*np.ones(len(LT1_x))

LT2_x = np.linspace(-50, 25, 26)
LT2_y = -13*np.ones(len(LT1_x))

DG2_x = []
DG2_y = []

ylines = [-17,-15,-13,-11,-9,-7]
for ii in ylines:
    for jj in np.arange(-25, 0):
        DG2_x.append(jj)
        DG2_y.append(ii)


fig1, ax1 = plt.subplots(1,1,num='grids')
ax1.plot(CT1_x,CT1_y, 'y.')
ax1.plot(LT1_x,LT1_y, 'y.')
ax1.plot(LT2_x,LT2_y, 'y.')
ax1.plot(DG2_x,DG2_y, 'y.')
ax1.invert_yaxis()
ax1.invert_xaxis()
# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

homechar = os.path.expanduser('~')
savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','surveys')
save_figures(savedn, 'survey_laydown', fig1)
