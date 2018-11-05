#!/usr/bin/env python

import DGS
import glob, os
from scipy.io import savemat
#import pickle
import csv
import os 
import errno

#os.chdir("/mnt/c/Projects/AdvocateBeach2018/data/raw/images/test/FoxPointBeach")
#os.chdir("/mnt/c/Projects/AdvocateBeach2018/data/raw/images/BeachSurveys/15_10_2018/PM/Longshore/m05_m/jnk/cropped")

date_str = "27_10_2018"
tide = "PM"
imset = "longshore1"

#C:\Projects\AdvocateBeach2018\data\processed\images\cropped
imdir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/images/cropped/beach_surveys/" + date_str + "/" + tide + "/" + imset
outdir = '/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/matfiles/'
outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/" + date_str + "/" + tide + "/" + imset

if not os.path.exists(outpydir):
    try:
        os.makedirs(outpydir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#for file in glob.glob(imdir + '*.jpg'):
for file in glob.glob(imdir + '/*-cropped.jpg'):

    image_file = file
    #'/home/tristan/Documents/Projects/DigitalGrainSizing/20180809_071614.jpg'

    density = 10 # process every 10 lines
    #resolution = 0.162 # mm/pixel [camera height 1]
    resolution = 0.123 # mm/pixel [camera height 2]
    dofilter = 1 # filter the imagei
    notes = 8 # notes per octave
    maxscale = 8 #Max scale as inverse fraction of data length
    verbose = 1 # print stuff to screen
    x = 0
    dgs_stats = DGS.dgs(image_file, density, resolution, dofilter, maxscale, notes, verbose, x)


    outpyfile = file[len(imdir):-4] + '.csv'
    # f = open(outpydir+outpyfile,"w")
    # pickle.dump(dgs_stats,f)
    # f.close()

    with open(outpydir+outpyfile, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dgs_stats.keys())
        w.writeheader()
        w.writerow(dgs_stats)

    # #remove spaces from var names for matlab compatability
    # dgs_stats['grain_size_bins'] = dgs_stats['grain size bins']
    # dgs_stats['grain_size_frequencies'] = dgs_stats['grain size frequencies']
    # dgs_stats['grain_size_skewness'] = dgs_stats['grain size skewness']
    # dgs_stats['grain_size_kurtosis'] = dgs_stats['grain size kurtosis']
    # dgs_stats['grain_size_sorting'] = dgs_stats['grain size sorting']
    # dgs_stats['mean_grain_size'] = dgs_stats['mean grain size']
    # del dgs_stats['grain size bins']
    # del dgs_stats['grain size frequencies']
    # del dgs_stats['grain size skewness']
    # del dgs_stats['grain size kurtosis']
    # del dgs_stats['grain size sorting']
    # del dgs_stats['mean grain size']
    #
    # # print data to .mat file
    # outfile = file[len(imdir):-4] + '.mat'
    # savemat(outdir+outfile, dgs_stats, oned_as='row')
