# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:00 2018

This script refines all DGS output (for each of many images) into easily callable
files. I.e. 1 file/grid_spec/tide.

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json


def build_gsize_arrays(gsizedir):
    '''Takes image-specific output from DGS package and consolidates into a single data file for each survey conponent, for each day
    '''

    allfn = sorted(os.listdir(gsizedir))

    mean_gsize = []
    sort = []
    skew = []
    kurt = []

    for fn in allfn:

        jnk = np.load(os.path.join(gsizedir, fn), encoding='latin1',  allow_pickle=True).item() # unpickle python 2 -> 3
        # jnk = np.load(os.path.join(gsizedir, fn), allow_pickle=True).item() # unpickle python

        mean_gsize.append(jnk['mean grain size'])
        sort.append(jnk['grain size sorting'])
        skew.append(jnk['grain size skewness'])
        kurt.append(jnk['grain size kurtosis'])

    # reshape the data into relevant grid
    # assumes I haven't missed any repeat/additional data points in my QC
    if gsizedir.split('/')[-1] == 'dense_array2' or gsizedir.split('/')[-1] == 'dense_array1':

        mean_gsize = np.array(mean_gsize).reshape(6, 24)
        sort = np.array(sort).reshape(6, 24)
        skew = np.array(mean_gsize).reshape(6, 24)
        kurt = np.array(sort).reshape(6, 24)

    else:

        mean_gsize = np.array(mean_gsize).reshape(len(mean_gsize),)
        sort = np.array(sort).reshape(len(mean_gsize),)
        skew = np.array(mean_gsize).reshape(len(mean_gsize),)
        kurt = np.array(sort).reshape(len(mean_gsize),)

    gsize = {"mean_grain_size": mean_gsize.tolist(), \
             "sorting": sort.tolist(), \
             "skewness": skew.tolist(), \
             "kurtosis": kurt.tolist()}

    return gsize


def main():

    # tide_range = range(28)
    tide_range = [14]

    ## load key:value dict to convert from yearday to tide num.
    #tidekeydn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "external", "tide_key_values.npy")
    #tidekey = np.load(tidekeydn).item()

    # for portability
    # homechar = "C:\\"
    homechar = os.path.expanduser("~") # linux

    grid_specs = ["longshore1"]
    # grid_spec = "longshore1"
    # grid_specs = ['cross_shore', 'longshore1', 'longshore2', 'dense_array1', 'dense_array2']

    for grid_spec in grid_specs:

        for ii in tide_range:

            tide = "tide" + str(ii)

            gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", "grainsize", "beach_surveys", tide, grid_spec)

            # if the survey doesn't exist for this day...
            if not os.path.exists(gsizedir):
                continue

            outdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                              "grainsize", "beach_surveys", tide, grid_spec + ".npy")

            outdir_json = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                              "grainsize", "beach_surveys", tide, grid_spec + ".json")

            gsize = build_gsize_arrays(gsizedir)

            np.save(outdir, gsize)

            with open(outdir_json, 'w') as fp:
                json.dump(gsize, fp)

#    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,7))
#    pos1 = ax1.imshow(gsize['mean_grain_size'], cmap='inferno')
#    fig.colorbar(pos1, ax=ax1)
#    pos2 = ax2.imshow(gsize['sorting'], cmap='inferno')
#    fig.colorbar(pos2, ax=ax2)

if __name__ == '__main__':
    main()
