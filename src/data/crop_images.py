#!/usr/bin/env python

import glob
from scipy.io import savemat
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio


def crop_image(img, wid, height, wid0, height0):
    """Crop an image.

    Takes original image and outputs new image cropped to desired width and
    height, around specified origin.

    newimg = crop_image(img, wid, height, wid0, height0)

    INPUTS
    img:    image to be cropped
    wid:    desired width as fraction of original image width (0-1 valued)
    height: desired height as fraction of original image width (0-1 valued)
    wid0, height0: origin of cropping window, if not centered (0-1).

    OUTPUT
    newimg: cropped image
    """

    # # testing
    # wid=2/3
    # height=2/3
    # wid0=1/6
    # height0=1/6
    # file = '/media/tristan2/Advocate2018_backup2/data/raw/images/OutdoorValidation/IMG_0206.JPG'
    # img = plt.imread(file)

    import numpy

    # find largest dimension
    arg_maxdim = numpy.argmax(img.shape)
    if arg_maxdim == 0:
        img_hgt = int(img.shape[1])
        img_wid = int(img.shape[arg_maxdim])
    else:
        img_hgt = int(img.shape[0])
        img_wid = int(img.shape[arg_maxdim])

    # mask dims
    h = int(img_hgt*height)
    w = int(img_wid*wid)

    # mask origin
    h0 = int(img_hgt*height0)
    w0 = int(img_wid*wid0)

    if arg_maxdim == 0:
        newimg = img[w0:w0+w, h0:h0+h, :]
    else:
        newimg = img[h0:h0+h, w0:w0+w, :]

    return newimg


homechar = os.path.expanduser("~") # linux

imset = 'longshore2' #'cross_shore' # longshore; longshore1; longshore2, dense_array1; dense_array2

# tide number - 1 being first of experiment (Sunday AM UTC)
# note 13 should really by tide B, but I'm sticking with my misinformed field convention
#{"15_10_2018_A.txt": 3, "16_10_2018_A.txt": 5, \
tide_table = {"17_10_2018_A": 7, "18_10_2018_A": 9, "19_10_2018_A": 11,"20_10_2018_A": 13, "21_10_2018_A": 14, "21_10_2018_B": 15,"22_10_2018_A": 16, "22_10_2018_B": 17, "23_10_2018_A": 18,"23_10_2018_B": 19, "24_10_2018_A": 20, "24_10_2018_B": 21,"25_10_2018_A": 22, "25_10_2018_B": 23, "26_10_2018_A": 24,"26_10_2018_B": 25, "27_10_2018_A": 26, "27_10_2018_B": 27}

#############################################################################
# FOR SURVEY IMAGES
notValidation = 1
# camera height was changed (at least once)
# use 1 for 1st setting, 2 for 2nd, etc
# 3 for validation
# cameraHeight = 2
date_str0 = ["17_10_2018", "18_10_2018", "19_10_2018", "20_10_2018", "21_10_2018", "21_10_2018", "22_10_2018", "22_10_2018", "23_10_2018", "23_10_2018", "24_10_2018", "24_10_2018", "25_10_2018", "25_10_2018", "26_10_2018", "26_10_2018", "27_10_2018", "27_10_2018"]
tide0 = ["AM", "AM", "AM", "AM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM", "AM", "PM"]
###############################################################################

################################################################################
## FOR PICAM IMAGES
# notValidation = 1
## camera height was changed (at least once)
## use 1 for 1st setting, 2 for 2nd, etc
## 3 for validation
# cameraHeight = 1
## date_str0 = ['23_10_2018'; '23_10_2018'; '24_10_2018'; '24_10_2018'; '25_10_2018'; '25_10_2018';...
##     '26_10_2018'; '26_10_2018'; '27_10_2018'; '27_10_2018']
## tide0 = ['AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM']
##############################################################################

#############################################################################
## FOR VALIDATION IMAGES
## camera height was changed (at least once)
## use 1 for 1st setting, 2 for 2nd, etc
## 3 for validation
# cameraHeight = 3
#notValidation = 0
#date_str0 = ['Oct21'; 'Oct21'; 'Oct21'; 'Oct21'; 'Oct21'; 'Oct21'; ...
#    'Oct25'; 'Oct25'; 'Oct25'; 'Oct25']
#tide0 = {'horn1'; 'horn2'; 'horn3'; 'bay1'; 'bay2'; 'bay3'; 'horn1'; 'horn2'; 'bay1'; 'bay2'}
################################################################################

# date_str0 = ['22_10_2018']
# tide0 = ['AM']

for kk in range(len(tide0)):

    # convert to tide number
    date_str = date_str0[kk]
    tide = tide0[kk];
    if tide == 'AM':
        sufx = '_A'
    else:
        sufx = '_B'
    tideid = date_str + sufx
    tidenum = 'tide' + str(tide_table[tideid])

    print(tidenum)

    # change camera height on the 18th (tide 9)
    if tide_table[tideid] < 9:
        cameraHeight = 1
    else:
        cameraHeight = 2


    dnin = os.path.join('/media','tristan2','Advocate2018_backup2','data',\
            'raw','images','BeachSurveys',date_str,tide,imset)

    if not os.path.exists(dnin):
        continue

    dnout = os.path.join('/media','tristan2','Advocate2018_backup2','data',\
            'processed','images','cropped','beach_surveys',tidenum,imset)


    ##########################################
    ## FOR PICAM IMAGES
    # dn = ['C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\tide15\pi74\']
    # dnout = ['C:\Projects\AdvocateBeach2018\data\processed\images\cropped\pi_cameras\tide15\pi74']
    ###########################################

    if not os.path.exists(dnout):
        try:
            os.makedirs(dnout)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    imnames = sorted(glob.glob(os.path.join(dnin, '*.JPG')))

    # check if this batch has already been processed
    alreadydone = sorted(glob.glob(os.path.join(dnout, '*.jpg')))

    if not alreadydone:

        counter = 0

        for imname in imnames:

            counter = counter + 1

            if counter%2 == 0:

                img = plt.imread(imname)

                if cameraHeight == 1:

                    # mask dims
                    height = 1/2
                    wid = 1/2

                    # mask origin
                    height0 = 1/4
                    wid0 = 1/4

                    # if strcmp(dn,'C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\tide27\pi74\')
                    # # avoid large water droplet on lens cover
                    #
                    #     # mask origin
                    #     h0 = floor(size(img, 1)/8*3)
                    #     w0 = floor(size(img, 2)/4)

                    # else

                elif cameraHeight == 2:

                    # mask dims
                    height = 2/3
                    wid = 2/3

                    # mask origin
                    height0 = 1/6
                    wid0 = 1/6

                # elif cameraHeight == 3: # validation
                #
                #     # mask dims
                #     height = floor(size(img, 1) - 2*1150)
                #     wid = floor(size(img, 2) - 2*1350)
                #
                #     # mask origin
                #     height0 = 1150;
                #     wid0 = 1350;

                newimg = crop_image(img, wid, height, wid0, height0)
                imageio.imwrite(os.path.join(dnout, imname[-12:-4] + '-cropped.jpg'), newimg)
