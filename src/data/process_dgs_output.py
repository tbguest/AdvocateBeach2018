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
import math


def build_gsize_arrays(gsizedir):
    '''Takes image-specific output from DGS package and consolidates into a single data file for each survey conponent, for each day
    '''

    allfn = sorted(os.listdir(gsizedir))

    mean_gsize = []
    sort = []
    skew = []
    kurt = []

    mean_gsize_geo = []
    sort_geo = []
    skew_geo = []
    kurt_geo = []

    mean_gsize_phi = []
    sort_phi = []
    skew_phi = []
    kurt_phi = []

    for fn in allfn:

        jnk = np.load(os.path.join(gsizedir, fn), encoding='latin1',  allow_pickle=True).item() # unpickle python 2 -> 3
        # jnk = np.load(os.path.join(gsizedir, fn), allow_pickle=True).item() # unpickle python

        ################## LATER ADDITION
        gsfreq = jnk['grain size frequencies']

        logmgs = []
        for elem in jnk['grain size bins']:
            logmgs.append(-math.log2(elem))

        # plt.figure(1)
        # plt.plot(logmgs, gsfreq,'.')
        # plt.plot(lnmgs, gsfreq,'.')

        # arithmetic
        mgs_a = np.sum(gsfreq*jnk['grain size bins'])
        sort_a = np.sqrt(np.sum(gsfreq*(jnk['grain size bins'] - mgs_a)**2))
        skew_a = np.sum(gsfreq*(jnk['grain size bins'] - mgs_a)**3)/sort_a**3
        kurt_a = np.sum(gsfreq*(jnk['grain size bins'] - mgs_a)**4)/sort_a**4

        # geometric
        mgs_g = np.exp(np.sum(gsfreq*np.log(jnk['grain size bins'])))
        sort_g = np.exp(np.sqrt(np.sum(gsfreq*(np.log(jnk['grain size bins']) - np.log(mgs_g))**2)))
        skew_g = np.sum(gsfreq*(np.log(jnk['grain size bins']) - np.log(mgs_g))**3)/np.log(sort_g)**3
        kurt_g = np.sum(gsfreq*(np.log(jnk['grain size bins']) - np.log(mgs_g))**4)/np.log(sort_g)**4

        # logarithmic (phi-scaled)
        mgs_p = np.sum(gsfreq*logmgs)
        sort_p = np.sqrt(np.sum(gsfreq*(logmgs - mgs_p)**2))
        skew_p = np.sum(gsfreq*(logmgs - mgs_p)**3)/sort_p**3
        kurt_p = np.sum(gsfreq*(logmgs - mgs_p)**4)/sort_p**4

        ##########################


        # mean_gsize.append(jnk['mean grain size'])
        # sort.append(jnk['grain size sorting'])
        # skew.append(jnk['grain size skewness'])
        # kurt.append(jnk['grain size kurtosis'])

        mean_gsize.append(mgs_a)
        sort.append(sort_a)
        skew.append(skew_a)
        kurt.append(kurt_a)

        mean_gsize_geo.append(mgs_g)
        sort_geo.append(sort_g)
        skew_geo.append(skew_g)
        kurt_geo.append(kurt_g)

        mean_gsize_phi.append(mgs_p)
        sort_phi.append(sort_p)
        skew_phi.append(skew_p)
        kurt_phi.append(kurt_p)

    # reshape the data into relevant grid
    # assumes I haven't missed any repeat/additional data points in my QC
    if gsizedir.split('/')[-1] == 'dense_array2' or gsizedir.split('/')[-1] == 'dense_array1':

        mean_gsize = np.array(mean_gsize).reshape(6, 24)
        sort = np.array(sort).reshape(6, 24)
        skew = np.array(skew).reshape(6, 24)
        kurt = np.array(kurt).reshape(6, 24)

        mean_gsize_geo = np.array(mean_gsize_geo).reshape(6, 24)
        sort_geo = np.array(sort_geo).reshape(6, 24)
        skew_geo = np.array(skew_geo).reshape(6, 24)
        kurt_geo = np.array(kurt_geo).reshape(6, 24)

        mean_gsize_phi = np.array(mean_gsize_phi).reshape(6, 24)
        sort_phi = np.array(sort_phi).reshape(6, 24)
        skew_phi = np.array(skew_phi).reshape(6, 24)
        kurt_phi = np.array(kurt_phi).reshape(6, 24)

    else:

        mean_gsize = np.array(mean_gsize).reshape(len(mean_gsize),)
        sort = np.array(sort).reshape(len(mean_gsize),)
        skew = np.array(mean_gsize).reshape(len(mean_gsize),)
        kurt = np.array(sort).reshape(len(mean_gsize),)

        mean_gsize_geo = np.array(mean_gsize_geo).reshape(len(mean_gsize_geo),)
        sort_geo = np.array(sort_geo).reshape(len(mean_gsize_geo),)
        skew_geo = np.array(skew_geo).reshape(len(mean_gsize_geo),)
        kurt_geo = np.array(kurt_geo).reshape(len(mean_gsize_geo),)

        mean_gsize_phi = np.array(mean_gsize_phi).reshape(len(mean_gsize_phi),)
        sort_phi = np.array(sort_phi).reshape(len(mean_gsize_phi),)
        skew_phi = np.array(skew_phi).reshape(len(mean_gsize_phi),)
        kurt_phi = np.array(kurt_phi).reshape(len(mean_gsize_phi),)

    gsize = {"mean_grain_size": mean_gsize.tolist(), \
             "sorting": sort.tolist(), \
             "skewness": skew.tolist(), \
             "kurtosis": kurt.tolist(), \
             "mean_grain_size_geo": mean_gsize_geo.tolist(), \
             "sorting_geo": sort_geo.tolist(), \
             "skewness_geo": skew_geo.tolist(), \
             "kurtosis_geo": kurt_geo.tolist(), \
             "mean_grain_size_phi": mean_gsize_phi.tolist(), \
             "sorting_phi": sort_phi.tolist(), \
             "skewness_phi": skew_phi.tolist(), \
             "kurtosis_phi": kurt_phi.tolist()}

    return gsize


def main():

    tide_range = range(28)
    # tide_range = [14]

    ## load key:value dict to convert from yearday to tide num.
    #tidekeydn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "external", "tide_key_values.npy")
    #tidekey = np.load(tidekeydn).item()

    # for portability
    # homechar = "C:\\"
    homechar = os.path.expanduser("~") # linux

    drivechar = '/media/tristan2/Advocate2018_backup2'

    # grid_specs = ["longshore1"]
    # grid_spec = "longshore1"
    grid_specs = ['cross_shore', 'longshore1', 'longshore2', 'dense_array1', 'dense_array2']

    for grid_spec in grid_specs:

        for ii in tide_range:

            tide = "tide" + str(ii)

            # gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", "grainsize", "beach_surveys", tide, grid_spec)

            # gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", "grainsize", "beach_surveys_reprocessed_x05", tide, grid_spec)
            gsizedir = os.path.join(drivechar, "data", "processed", "grainsize", "beach_surveys_reprocessed_x08", tide, grid_spec)

            # if the survey doesn't exist for this day...
            if not os.path.exists(gsizedir):
                continue

            # outdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
            #                   "grainsize", "beach_surveys", tide, grid_spec + ".npy")
            #
            # outdir_json = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
            #                   "grainsize", "beach_surveys", tide, grid_spec + ".json")

            # outdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
            #                   "grainsize", "beach_surveys_reprocessed_x05", tide)

            outdir = os.path.join(drivechar, "data", "processed", \
                              "grainsize", "beach_surveys_reprocessed_x08", tide)

            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            outfn = os.path.join(outdir, grid_spec + ".npy")

            outfn_json = os.path.join(outdir, grid_spec + ".json")

            gsize = build_gsize_arrays(gsizedir)

            np.save(outfn, gsize)

            with open(outfn_json, 'w') as fp:
                json.dump(gsize, fp)

#    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,7))
#    pos1 = ax1.imshow(gsize['mean_grain_size'], cmap='inferno')
#    fig.colorbar(pos1, ax=ax1)
#    pos2 = ax2.imshow(gsize['sorting'], cmap='inferno')
#    fig.colorbar(pos2, ax=ax2)

if __name__ == '__main__':
    main()
