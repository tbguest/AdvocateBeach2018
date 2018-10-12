#!/usr/bin/env python

import DGS
import glob, os
from scipy.io import savemat

os.chdir("/mnt/c/Projects/AdvocateBeach2018/data/raw/images/test/FoxPointBeach")
for file in glob.glob("*.jpg"):
    image_file = '/mnt/c/Projects/AdvocateBeach2018/data/raw/images/test/FoxPointBeach/' + file
    #'/home/tristan/Documents/Projects/DigitalGrainSizing/20180809_071614.jpg'

    density = 10 # process every 10 lines
    resolution = 0.25 # mm/pixel
    dofilter =1 # filter the imagei
    notes = 8 # notes per octave
    maxscale = 8 #Max scale as inverse fraction of data length
    verbose = 1 # print stuff to screen
    x = 0
    dgs_stats = DGS.dgs(image_file, density, resolution, dofilter, maxscale, notes, verbose, x)

    #remove spaces from var names for matlab compatability
    dgs_stats['grain_size_bins'] = dgs_stats['grain size bins']
    dgs_stats['grain_size_frequencies'] = dgs_stats['grain size frequencies']
    dgs_stats['grain_size_skewness'] = dgs_stats['grain size skewness']
    dgs_stats['grain_size_kurtosis'] = dgs_stats['grain size kurtosis']
    dgs_stats['grain_size_sorting'] = dgs_stats['grain size sorting']
    dgs_stats['mean_grain_size'] = dgs_stats['mean grain size']
    del dgs_stats['grain size bins']
    del dgs_stats['grain size frequencies']
    del dgs_stats['grain size skewness']
    del dgs_stats['grain size kurtosis']
    del dgs_stats['grain size sorting']
    del dgs_stats['mean grain size']

    # print data to .mat file
    outfile = file[0:-4] + '.mat'
    savemat(outfile, dgs_stats, oned_as='row')
