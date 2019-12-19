#!/usr/bin/env python

import DGS
#import pickle
import os
import errno
import numpy as np
import glob


drivechar = "/mnt/g"


def imname2scalefactor(image_file, tide, pinum):

    # img has form: img**unixtime-fractionalsecond**-cropped.jpg
    # use this to extract time so the range can be refernced to the sonar data

#    img = "img1540307172-338723-cropped.jpg" #img1540301878-322653
#    image_file = "img1540304182-837550-cropped.jpg" # from pi71 on tide19

    usec = image_file[-29:-19]
    ufrac = image_file[-18:-12]
    utime = usec + "." + ufrac

    # homechar = "/mnt/c"
    # homechar = os.path.expanduser('~')

    # rangefile = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
           # "range_data", "bed_level", "tide" + tide, "sonar" + pinum + ".npy")

    rangefile = os.path.join(drivechar, "data", "interim", \
           "range_data", "bed_level", "tide" + tide, "sonar" + pinum + ".npy")

    # rangefile = homechar + "/Projects/AdvocateBeach2018/data/interim/range_data/bed_level/tide" + tide + "/sonar" + pinum + ".npy"

    # for troubleshooting:
#    homechar = "C:\\"
#    rangefile = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
#                "range_data", "bed_level", "tide" + tide, "sonar" + pinum + ".npy")

    ##
    vvv = np.load(rangefile, allow_pickle=True).item()
    smth = vvv['smoothed chunks']
    rtime = smth[0]
    sonar_range = smth[1]
    ##

#    vvv = np.load(rangefile).item()
#    rtime = vvv['time']
#    sonar_range = vvv['bed level']

    Ivalidrange = np.argmin(abs(float(utime) + 60*60*3 - rtime)) # adjust for ADT->UTC
    t_resid = np.abs(float(utime) + 60*60*3 - rtime[Ivalidrange])

    if t_resid > 60:
        return 0

    offset = 75 # [mm] must be measured from physical array setup
    pi_range = sonar_range[Ivalidrange] + offset

#     - Max resolution: 3280 x 2464
#  - fov: 62.2 x 48.8 deg

    ground_width = 2*pi_range*np.tan(62.2/2*np.pi/180)
    mm_per_pix = ground_width/3280

#    # check for consistency with other dimension
#    ground_width2 = 2*pi_range*np.tan(48.8/2*np.pi/180)
#    mm_per_pix2 = ground_width2/2464

    return mm_per_pix




homechar = os.path.expanduser('~')

# date_str0 = "21_10_2018"
tide = "19"
# tide = "27"

# pinums = ['71', '73', '74']
pinums = ['71', '72','73', '74']

for pinum in pinums:
    # pinum = "74"

    #imdir = "/mnt/c/Projects/AdvocateBeach2018/data/interim/images/PiCameras/" + date_str0 + "/horn_growth/selected/cropped"
    #imdir = "/mnt/c/Projects/AdvocateBeach2018/data/interim/images/PiCameras/tide" + tide + "/pi" + pinum + "/cropped"
    #C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\tide19\pi71\cropped

    # imdir = "/mnt/c/Projects/AdvocateBeach2018/data/processed/images/cropped/pi_cameras/tide" + tide + "/pi" + pinum
    # outdir = "/mnt/c/Projects/AdvocateBeach2018/data/interim/grainsize_dists/pi_array/tide" + tide + "/pi" + pinum + "/smooth_bed_level/"

    # imdir = os.path.join(homechar,'Projects','AdvocateBeach2018','data','processed',\
    #                         'images','cropped','pi_cameras','tide'+tide,'pi'+pinum)
    imdir = os.path.join(drivechar,'data','processed',\
                            'images','cropped','pi_cameras','tide'+tide,'pi'+pinum)
    # outdir = os.path.join(homechar,'Projects','AdvocateBeach2018','data','interim',\
    #                         'grainsize','pi_array','tide'+tide,'pi'+pinum,'smooth_bed_level')
    outdir = os.path.join(drivechar,'data','interim',\
                            'grainsize','pi_array','tide'+tide,'pi'+pinum,'smooth_bed_level_reprocessed_x15')

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    #for file in glob.glob(imdir + '*.jpg'):
    for file in glob.glob(os.path.join(imdir, '*-cropped.jpg')):

        image_file = file

    	  #'/home/tristan/Documents/Projects/DigitalGrainSizing/20180809_071614.jpg'

        mm_per_pix = imname2scalefactor(image_file, tide, pinum)

        if mm_per_pix is 0:
            continue

        density = 10 # process every 10 lines
    	  #resolution = 0.162 # mm/pixel [camera height 1]
    #        resolution = 0.123 # mm/pixel [camera height 2]
        resolution = mm_per_pix # mm/pixel [camera height 2]
        dofilter = 1 # filter the imagei
        notes = 8 # notes per octave
        maxscale = 8 #Max scale as inverse fraction of data length
        verbose = 0 # print stuff to screen
        x = 1.5
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

        np.save(outdir + file[len(imdir):-4] + '.npy', dgs_stats)

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
