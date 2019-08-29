#!/usr/bin/env python

import DGS
import glob
import os
import errno
import numpy as np


# os.chdir('/home/tristan2/code/pyDGS/DGS')

homechar = os.path.expanduser("~") # linux

# sample date and location
date_str0 = ["Oct21", "Oct21", "Oct21", "Oct21", "Oct21", \
             "Oct21", "Oct25", "Oct25", "Oct25", "Oct25"]
tide0 = ["_bay1", "_bay2", "_bay3", "_horn1", "_horn2", "_horn3", \
         "_bay1", "_bay2", "_horn1", "_horn2"]

for nn in range(len(tide0)):

    imgdir = os.path.join("/media","tristan2","Advocate2018_backup2", "data", "raw", \
                    "images", "OutdoorValidation", date_str0[nn] + tide0[nn])

    allfiles = sorted(glob.glob(os.path.join(imgdir,'*cropped.jpg')))

    outdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data',\
            'processed','grainsize','validation', 'OutdoorValidation_x0_maxscale5', date_str0[nn] + tide0[nn])

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    for file in sorted(glob.glob(os.path.join(imgdir,'*-cropped.jpg'))):

        image_file = file

        # change camera height on the 18th (tide 9)
        resolution = 0.13
        density = 10 # process every 10 lines
        dofilter = 1 # filter the imagei
        notes = 8 # notes per octave
        maxscale = 5 #Max scale as inverse fraction of data length
        verbose = 0 # print stuff to screen
        x = 0
        dgs_stats = DGS.dgs(image_file, density, resolution, dofilter, maxscale, notes, verbose, x)

        np.save(outdir + file[len(imgdir):-4] + '.npy', dgs_stats)
