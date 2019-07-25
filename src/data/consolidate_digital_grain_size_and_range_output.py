# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:34:20 2019

@author: Tristan Guest

Use this script to organize range and grain size data (separately) into
separate chunk files with data from all pimodules grouped together in a single
dict.

17 April 2019:
Modified to also save raw range data chunks for swash analysis.
"""

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


def make_gs_chunk(mgs, std, tvec, tt_start, tt_end):
    ''' Uses the predefined points of interest (POI) to break the full record
    of mean grain size and sorting into the chunks associated with each
    sampling location.

    Returns dict of gs data from all pis for given chunk
    '''

    # initialize dict
    chunkdata_gs = {}

    for Ipi in np.sort(list(mgs.keys())):

        Ichunk = np.where(np.logical_and(tvec[Ipi]>=tt_start, tvec[Ipi]<=tt_end))
        data = {'tvec': tvec[Ipi][Ichunk], 'mgs': mgs[Ipi][Ichunk], 'sort': std[Ipi][Ichunk]}
        chunkdata_gs[Ipi] = data

    return chunkdata_gs

def make_range_chunk(rng, tvec_rng, raw, tvec_raw, tt_start, tt_end):
    ''' Uses the predefined points of interest (POI) to break the full range
    record into the chunks associated with each sampling location.

    Returns dict of range data (bed level only; raw range) from all pis for given chunk
    '''

    # initialize dict
    chunkdata_rng = {}

    for Ipi in np.sort(list(rng.keys())):

        Ichunk_rng = np.where(np.logical_and(tvec_rng[Ipi]>=tt_start, tvec_rng[Ipi]<=tt_end))
        Ichunk_raw = np.where(np.logical_and(tvec_raw[Ipi]>=tt_start, tvec_raw[Ipi]<=tt_end))
        data = {'tvec': tvec_rng[Ipi][Ichunk_rng], 'range': rng[Ipi][Ichunk_rng], \
                'tvec_raw': tvec_raw[Ipi][Ichunk_raw], 'raw_range': raw[Ipi][Ichunk_raw]}
        chunkdata_rng[Ipi] = data

    return chunkdata_rng


# homechar = "C:\\"
homechar = os.path.expanduser('~')

# # tide 15
# tide = '15'
# skip_first_chunk = 1 # my attempt to reconcile GPS positions with apparent range data positions
# skip_last_chunk = 0

# tide 19
tide = '19'
skip_first_chunk = 1 # my attempt to reconcile GPS positions with apparent range data positions
skip_last_chunk = 0

# tide 21 ?

# # tide 27
# tide = '27'
# skip_first_chunk = 0 # my attempt to reconcile GPS positions with apparent range data positions
# skip_last_chunk = 1

pinums = ['71', '72', '73', '74']

# where did this come from?
timesdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                        "interim", "range_data", "start_and_end_times", \
                        "tide" + tide + ".npy")

# where did this come from?
POI = np.load(timesdir)

nchunks = int(len(POI)/2)

mgs_matrix = np.empty([0])
sort_matrix = np.empty([0])

# populate dict of data from all 4 sensors
mgs = {}
std = {}
tvec = {}

rng = {}
raw = {}
tvec_raw = {}
tvec_rng = {}


for pinum in pinums:
#pinum = '72'

    ### grain size: ###

    # gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
               # "grainsize", "pi_array", "tide" + tide, 'pi' + pinum)
    gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                "grainsize", "pi_array", "tide" + tide, 'pi' + pinum, "smooth_bed_level")

    mean_gs = []
    std_gs = []
    timg = []

    # for tide 27, where no file exists for pi72
    if not os.path.exists(gsizedir):
        continue

    for file in sorted(glob(os.path.join(gsizedir, 'img*.npy'))):

        foo = np.load(file, allow_pickle=True, encoding='latin1').item()
        timg.append(float(file[-29:-19] + '.' + file[-18:-12]))
        mean_gs.append(foo['mean grain size'])
        std_gs.append(foo['grain size sorting'])

# plt.plot(timg, mean_gs, 'o')

    tt_img = np.array(timg) + 3.0*60*60
    mgs_img = np.array(mean_gs)
    std_img = np.array(std_gs)

# plt.plot(tt_img, mgs_img, 'o')

    dictkey = 'pi' + pinum
    # populate dict of data from all 4 sensors:
    mgs[dictkey] = mgs_img
    std[dictkey] = std_img
    tvec[dictkey] = tt_img

# plt.plot(tvec['pi71'],mgs['pi71'], 'o')

    ### range data: ###

    rangedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                "range_data", "bed_level", "tide" + tide)

    # skirts the issue of missing range data from broken sd card
    if not os.path.exists(os.path.join(rangedir, 'sonar' + pinum + '.npy')):
        continue

    jnk = np.load(os.path.join(rangedir, 'sonar' + pinum + '.npy'), allow_pickle=True).item()

    raw[dictkey] = jnk['raw range'][1]
    tvec_raw[dictkey] = jnk['raw range'][0]
    rng[dictkey] = jnk['raw bed level'][1]
    tvec_rng[dictkey] = jnk['raw bed level'][0]


# cycle through chunks, create chunk-specific data dicts
counter = 0
for jj in range(0 + skip_first_chunk, nchunks - skip_last_chunk):

    counter += 1

    tt_start = POI[jj*2][0]
    tt_end = POI[1 + jj*2][0]

    chunkdata_gs = make_gs_chunk(mgs, std, tvec, tt_start, tt_end)
    chunkdata_rng = make_range_chunk(rng, tvec_rng, raw, tvec_raw, tt_start, tt_end)

    savedir_gs = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'grainsize', 'pi_array', 'tide' + tide)

    savedir_rng = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'tide' + tide)

    if not os.path.exists(savedir_gs):
        try:
            os.makedirs(savedir_gs)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if not os.path.exists(savedir_rng):
        try:
            os.makedirs(savedir_rng)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # np.save(os.path.join(savedir_gs, 'chunk' + str(jj+1) + '.npy'), chunkdata_gs)
    # np.save(os.path.join(savedir_rng, 'chunk' + str(jj+1) + '.npy'), chunkdata_rng)

    np.save(os.path.join(savedir_gs, 'chunk' + str(counter) + '.npy'), chunkdata_gs)
    np.save(os.path.join(savedir_rng, 'chunk' + str(counter) + '.npy'), chunkdata_rng)
