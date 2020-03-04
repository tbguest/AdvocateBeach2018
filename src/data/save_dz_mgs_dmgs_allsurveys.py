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

# %matplotlib qt5

plt.close('all')



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



saveFlag = 1

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux
drivechar = '/media/tristan2/Advocate2018_backup2'

# dn_out = os.path.join(homechar,'Projects','AdvocateBeach2018','data', 'processed', 'survey_data', 'reprocessed')
# dn_out = os.path.join(homechar,'Projects','AdvocateBeach2018','data', 'processed', 'survey_data', 'reprocessed_x10')
dn_out = os.path.join(drivechar,'data', 'processed', 'survey_data', 'reprocessed_x08')

# cross-shore HWL coord and tide index
hwl0 = [-21,-9,-15,-15,-18,-15,-15,-15,-18,-21,-18,-18,-18,-18]
# hwl0 = [-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27]
Ihwl = [14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26, 27]

np.mean(hwl0)
np.std(hwl0)

# grid_specs = ["cross_shore"]#["longshore1", "longshore2"]#, "dense_array2"]
# grid_specs = ["cross_shore","longshore1", "dense_array2"]
grid_specs = ["cross_shore","longshore1", "longshore2", "dense_array2"]
# grid_specs = ["longshore1"]#, "longshore1", "longshore2", "dense_array2"]
# grid_specs = ["longshore1"]

# guess at cross-shore profile length
npts = 150

for grid_spec in grid_specs:
# grid_spec = ["dense_array2"]#, "longshore1", "longshore2", "dense_array2"]

    # for tide comparison with wave data
    dz_tides = {}
    mgs_tides = {}
    dmgs_tides = {}

    # # wave data
    Hs = {}
    Tp = {}
    steepness = {}
    iribarren = {}
    wave_energy = {}
    wave_energy_wind = {}
    wave_energy_swell = {}
    maxdepth = {}
    hwl = {}

    # grid_spec = "longshore2"
    # grid_spec = "longshore1"
    # grid_spec = "cross_shore"
    # grid_spec = "dense_array2"

    if grid_spec == 'cross_shore':
        start_tide = 13
    elif grid_spec == 'longshore1':
        # start_tide = 15
        start_tide = 13
    else:
        start_tide = 14

    tide_range = range(start_tide, 28)
    tide_axis = np.arange(start_tide+1,28) # for plotting later

    counter = -1

    for ii in tide_range:

        counter = counter + 1

        # so tide sets of points can be saved individually
        dz_tides_tmp = []
        mgs_tides_tmp = []
        dmgs_tides_tmp = []

        tide = "tide" + str(ii)

        # gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
        #                       "grainsize", "beach_surveys", tide, grid_spec + ".json")

        # gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
        #                       "grainsize", "beach_surveys_reprocessed_x10", tide, grid_spec + ".json")

        gsizefn = os.path.join(drivechar, "data", "processed", \
                              "grainsize", "beach_surveys_reprocessed_x08", tide, grid_spec + ".json")

        gpsfn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                              "GPS", "by_tide", tide, grid_spec + ".json")

        wavefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                      "pressure", "wavestats", tide + ".json")

        # wave data
        if not os.path.exists(wavefn):
            continue

        # GPS data
        if not os.path.exists(gpsfn):
            continue

        # grainsize data
        if not os.path.exists(gsizefn):
            continue

        # if this hasn't already been done for different grid_spec
        if not ii in Hs:

            with open(wavefn, 'r') as fpp:
                bar = json.load(fpp)

            Hs[ii] = np.mean(np.array(bar["Hs"]))
            Tp[ii] = np.mean(np.array(bar["Tp"]))
            steepness[ii] = np.mean(np.array(bar["steepness"]))
            iribarren[ii] = np.mean(np.array(bar["Iribarren"]))
            wave_energy[ii] = np.mean(np.array(bar["wave_energy"]))
            wave_energy_wind[ii] = np.mean(np.array(bar["wave_energy_wind"]))
            wave_energy_swell[ii] = np.mean(np.array(bar["wave_energy_swell"]))
            maxdepth[ii] = np.max(bar["depth"])
            if ii in Ihwl:
                hwl[ii] = hwl0[counter-1]

        # sediment data
        # jnk = np.load(gsizefn, allow_pickle=True).item()
        with open(gsizefn, 'r') as fpp:
            jnk = json.load(fpp)

        mgs0 = np.array(jnk['mean_grain_size'])

        if grid_spec == 'dense_array2':
            mgs0 = np.squeeze(np.reshape(np.array(jnk['mean_grain_size']), (1,mgs0.shape[0]*mgs0.shape[1])))

        # np.reshape(mgs0, (1,mgs0.shape[0]*mgs0.shape[1]))

        mgs = np.pad(mgs0, (0, npts-len(mgs0)), 'constant', constant_values=(np.nan,np.nan))

        # foo = np.load(gpsfn, allow_pickle=True).item()
        with open(gpsfn, 'r') as fpp:
            foo = json.load(fpp)

        # x = np.array(foo['x']).reshape(6, 24)
        # y = np.array(foo['y']).reshape(6, 24)
        # z = np.array(foo['z']).reshape(6, 24)
        x0 = np.array(foo['x'])
        y0 = np.array(foo['y'])
        z0 = np.array(foo['z'])

        x = np.pad(x0, (0, npts-len(x0)), 'constant', constant_values=(np.nan,np.nan))
        yjunk = np.pad(y0, (0, npts-len(y0)), 'constant', constant_values=(np.nan,np.nan))
        y = np.around(yjunk) # since gps values aren't exact.
        z = np.pad(z0, (0, npts-len(z0)), 'constant', constant_values=(np.nan,np.nan))

        # for first iteration - no data logging yet
        if counter == 0:

            last_z = np.copy(z)
            dz = z - last_z

            last_mgs = np.copy(mgs)
            dmgs = mgs - last_mgs

        else:

            dz = z - last_z
            dmgs = mgs - last_mgs

            if grid_spec == 'cross_shore':
                maxIy = 12 # deals with different length profiles
            else:
                maxIy = len(y)

            # populate bin based on cross-shore beach region (rel. to HWL)
            for yi in range(maxIy):

                if ~np.isnan(y[yi]):

                    dz_tides_tmp.append(dz[yi])
                    mgs_tides_tmp.append(mgs[yi])
                    dmgs_tides_tmp.append(dmgs[yi])

            if ii not in dz_tides:
                dz_tides[ii] = dz_tides_tmp
                mgs_tides[ii] = mgs_tides_tmp
                dmgs_tides[ii] = dmgs_tides_tmp
            else:
                dz_tides[ii].extend(dz_tides_tmp)
                mgs_tides[ii].extend(mgs_tides_tmp)
                dmgs_tides[ii].extend(dmgs_tides_tmp)

            # update
            last_z = np.copy(z)
            last_mgs = np.copy(mgs)
            last_dmgs = np.copy(dmgs)


    msd_data = {'dz': dz_tides, 'mgs': mgs_tides, 'dmgs': dmgs_tides, 'Hs': Hs, 'Tp': Tp, \
                'steepness': steepness, 'iribarren': iribarren, 'wave_energy': wave_energy, \
                'maxdepth': maxdepth, 'hwl': hwl, 'x':x[:len(dz_tides[ii])], 'y':y[:len(dz_tides[ii])]}


    saveFlag = 0
    if saveFlag == 1:

        fn_out = os.path.join(dn_out, grid_spec + '.npy')

        if not os.path.exists(dn_out):
            try:
                os.makedirs(dn_out)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        np.save(fn_out, msd_data)
