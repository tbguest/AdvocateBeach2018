#!/usr/bin/env python

import DGS
import glob
import os
import errno
import numpy as np


# os.chdir('/home/tristan2/code/pyDGS/DGS')

homechar = os.path.expanduser("~") # linux
homechar = "C:\\"

# drive dir
drivechar = "/mnt/g"

# imset = "cross_shore"
# imset = "longshore2"
# imset = "longshore1"
# imset = "dense_array2"

imsets = ["cross_shore","longshore2", "longshore1", "dense_array2"]

for imset in imsets:

    tides = range(28)
    # tides = [14]

    for n in tides:

        tidenum = 'tide' + str(n)

        # imdir = os.path.join('/media','tristan2','Advocate2018_backup2','data',\
                # 'processed','images','cropped','beach_surveys',tidenum,imset)

        imdir = os.path.join(drivechar,'data',\
                'processed','images','cropped','beach_surveys',tidenum,imset)

        # move on if the input imgs don't exist
        if not os.path.exists(imdir):
            continue

        # outdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data',\
        #         'processed','grainsize','beach_surveys',tidenum,imset)

        outdir = os.path.join(drivechar,'data',\
                'processed','grainsize','beach_surveys_reprocessed_x10',tidenum,imset)

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        for file in sorted(glob.glob(os.path.join(imdir,'*-cropped.jpg'))):

            image_file = file

            # change camera height on the 18th (tide 9)
            if n < 9:
                resolution = 0.162
            else:
                resolution = 0.13
            density = 10 # process every 10 lines
            dofilter = 1 # filter the imagei
            notes = 8 # notes per octave
            maxscale = 8 #Max scale as inverse fraction of data length
            verbose = 0 # print stuff to screen
            x = 1.0
            dgs_stats = DGS.dgs(image_file, density, resolution, dofilter, maxscale, notes, verbose, x)

            np.save(outdir + file[len(imdir):-4] + '.npy', dgs_stats)
