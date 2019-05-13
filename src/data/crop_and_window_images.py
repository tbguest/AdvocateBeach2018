#!/usr/bin/env python

import glob, os
from scipy.io import savemat
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio

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

        # imdir = "/mnt/c/Projects/AdvocateBeach2018/data/raw/images/LabValidation/" + date_str + tide
        # imdir = "/mnt/c/Projects/AdvocateBeach2018/data/raw/images/OutdoorValidation/" + date_str + tide
        # outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/LabValidation_xmin05/"+ date_str + tide
        # outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/OutdoorValidation_xmin05_ms5p5/"+ date_str + tide

        imdir = os.path.join('C:\\', 'Projects', 'AdvocateBeach2018', 'data', \
                'raw', 'images', 'OutdoorValidation',  date_str + tide)

    # else:
    #
    #     #C:\Projects\AdvocateBeach2018\data\processed\images\cropped
    #     imdir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/images/cropped/beach_surveys/" + date_str + "/" + tide + "/" + imset
    #     outdir = '/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/matfiles/'
    #     #	outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/" + date_str + "/" + tide + "/" + imset
    #     outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/" + tidedir + "/" + imset
    # #    outpydir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/grainsize_dists/json/" + tidedir + "/" + imset


	 #for file in glob.glob(imdir + '*.jpg'):
    for file in glob.glob(imdir + '/*-cropped.jpg'):

        # file = imdir + '\\IMG_0144-cropped.jpg'

        newdn = os.path.join(imdir, file[:-4])
        if not os.path.exists(newdn):
            try:
                os.makedirs(newdn)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        window_images = True

        if window_images is True:

            im = plt.imread(file)

            # find largest dimension
            maxdim = np.max(im.shape)
            arg_maxdim = np.argmax(im.shape)

            # window and overlap
            npieces = 3 # chunks to average
            window = (maxdim/2).astype(int)
            overlap = np.floor(0.5*window) # 50 pc overlap

            for win in range(npieces):

                # make sure to operate on largest image dimension
                if arg_maxdim is 0:
                    imchunk = im[int(win*overlap) : int(win*overlap+window), :, :]
                else:
                    imchunk = im[:, int(win*overlap) : int(win*overlap+window), :]

                imageio.imwrite(newdn + '/' + 'chunk_' + str(win) + '.jpg', imchunk)
