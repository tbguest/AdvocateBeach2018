#!/usr/bin/env python

import DGS
import glob, os
from scipy.io import savemat
#import pickle
import csv
import os 
import errno
import numpy as np
import json

#os.chdir("/mnt/c/Projects/AdvocateBeach2018/data/raw/images/test/FoxPointBeach")
#os.chdir("/mnt/c/Projects/AdvocateBeach2018/data/raw/images/BeachSurveys/15_10_2018/PM/Longshore/m05_m/jnk/cropped")

validationFlag = 1

# tide number - 1 being first of experiment (Sunday AM UTC)
# note 13 should really by tide B, but I'm sticking with my misinformed field convention 
#{"15_10_2018_A.txt": 3, "16_10_2018_A.txt": 5, \
tide_table = {"17_10_2018_A": 7, "18_10_2018_A": 9, "19_10_2018_A": 11, "20_10_2018_A": 13, \
    "21_10_2018_A": 14, "21_10_2018_B": 15, "22_10_2018_A": 16, "22_10_2018_B": 17, "23_10_2018_A": 18, \
    "23_10_2018_B": 19, "24_10_2018_A": 20, "24_10_2018_B": 21, "25_10_2018_A": 22, \
    "25_10_2018_B": 23, "26_10_2018_A": 24, "26_10_2018_B": 25, "27_10_2018_A": 26, \
    "27_10_2018_B": 27}

if validationFlag is 1:
    
    date_str0 = ["Oct21", "Oct21", "Oct21", "Oct21", "Oct21", \
                 "Oct21", "Oct25", "Oct25", "Oct25", "Oct25"]
    tide0 = ["_bay1", "_bay2", "_bay3", "_horn1", "_horn2", "_horn3", \
             "_bay1", "_bay2", "_horn1", "_horn2"]    
    
else:
    
    date_str0 = ["21_10_2018", "21_10_2018", "22_10_2018", "22_10_2018", "23_10_2018", \
                 "23_10_2018", "24_10_2018", "24_10_2018", "25_10_2018", "25_10_2018", \
                 "26_10_2018", "26_10_2018", "27_10_2018", "27_10_2018"]
    tide0 = ["AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM"]    

for n in range(0, len(tide0)):
    
    if validationFlag is 1:
        
        date_str = date_str0[n]
        tide = tide0[n]
        
        imdir = "/mnt/c/Projects/AdvocateBeach2018/data/raw/images/LabValidation/" + date_str + tide
        outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/LabValidation_xmin05/"+ date_str + tide

    else: 

        date_str = date_str0[n]
        tide = tide0[n]
        if tide == "AM":
            tidedir = "tide" + str(tide_table[date_str + "_A"])
        else:
            tidedir = "tide" + str(tide_table[date_str + "_B"])
       
        imset = "dense_array2"
    
        #C:\Projects\AdvocateBeach2018\data\processed\images\cropped
        imdir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/images/cropped/beach_surveys/" + date_str + "/" + tide + "/" + imset
        outdir = '/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/matfiles/'
        #	outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/" + date_str + "/" + tide + "/" + imset
        outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/" + tidedir + "/" + imset
    #    outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/json/" + tidedir + "/" + imset

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
#        resolution = 0.123 # mm/pixel [camera height 2]
        resolution = 0.13 # mm/pixel [camera height 2]
        dofilter = 1 # filter the imagei
        notes = 8 # notes per octave
        maxscale = 8 #Max scale as inverse fraction of data length
        verbose = 0 # print stuff to screen
        x = -0.5
        dgs_stats = DGS.dgs(image_file, density, resolution, dofilter, maxscale, notes, verbose, x)


        outpyfile = file[len(imdir):-4] + '.csv'
		  # f = open(outpydir+outpyfile,"w")
		  # pickle.dump(dgs_stats,f)
		  # f.close()

#        with open(outpydir+outpyfile, 'wb') as f:  # Just use 'w' mode in 3.x
#            w = csv.DictWriter(f, dgs_stats.keys())
#            w.writeheader()
#            w.writerow(dgs_stats)
            

#        with open(outpydir + file[len(imdir):-4] + '.json', 'w') as f:
#            json.dump(dgs_stats, f)    
            
        np.save(outpydir + file[len(imdir):-4] + '.npy', dgs_stats)      

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
