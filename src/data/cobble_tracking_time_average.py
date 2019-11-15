# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:49:57 2019

@author: Tristan Guest
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import errno
import cv2 as cv
# %matplotlib qt5


 # ffmpeg -r 25 -f image2 -start_number 6830 -i img%06d.jpg time avg_test.webm
 # ffmpeg -r 25 -f image2 -start_number 710 -i img%06d.jpg position4_10frameavg.webm


homechar = "C:\\"
ext_drive = "F:\\"
use_ext_drive = 1 # 0 or 1: don't or do use ext drive

tide = "tide15"
# tide = "tide19"
position = "position2"

if tide == "tide19":
    # vidspec = "vid_1540304255" # pos1
    vidspec = "vid_1540307860" # pos2
    # vidspec = "vid_1540307860" # pos3
    # vidspec = "vid_1540311466" # pos4
elif tide == "tide15":
    # vidspec = "vid_1540125835" # pos1
    vidspec = "vid_1540129441" # pos2
    # vidspec = "vid_1540129441" # pos3

if use_ext_drive == 1:
    savedir = os.path.join(ext_drive, "data", "interim", \
                               "images","timeAverage", tide, position, vidspec, 'tenframeaverage')
    imgdir = os.path.join(ext_drive, "data", "interim", \
                          "images", "fromVideo", tide, position, vidspec)
else:
    savedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                               "images","timeAverage", tide, position, vidspec, 'tenframeaverage')
    imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                          "images", "fromVideo", tide, position, vidspec)

imgs = sorted(glob.glob(os.path.join(imgdir, 'img*.jpg')))

red_range = [175,255]
grn_range = [175,255]
blu_range = [175,255]

# main loop
im0 = 0
counter = -1
N = 10

while 1:

    counter += 1
    im_avg = np.zeros(np.shape(plt.imread(imgs[0])))

    if not os.path.isfile(imgs[im0 + counter + N-1]):
        break

    # for file in imgs:
    for i in (im0 + counter + np.arange(N)):

        file = imgs[i]
        im = plt.imread(file)
        # im = plt.imread(imgs[504])

        r, g, b = cv.split(im)

        is_blu = cv.inRange(b, blu_range[0], blu_range[1])
        is_grn = cv.inRange(g, grn_range[0], grn_range[1])
        is_red = cv.inRange(r, red_range[0], red_range[1])
        wht_mask = np.logical_and(is_red, is_grn)
        wht_mask = np.logical_and(wht_mask, is_blu)

        nowhite = im.copy()
        # nowhite = newimg[]
        nowhite[wht_mask!=0] = (0,0,0)

        im_avg = im_avg + nowhite

        # plt.figure(1).clf()
        # plt.imshow(nowhite)
        # plt.tight_layout()
        # plt.draw()
        # plt.show()

    # plt.figure(1).clf()
    # plt.imshow((im_avg/N).astype('uint8'))
    # plt.tight_layout()
    # plt.show()

    mean_img = (im_avg/N).astype('uint8')

    fn = file[-13:-4] + '.jpg'

    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.imsave(os.path.join(savedir, fn), mean_img)

        # cv.imwrite(os.path.join(savedir, fn), close)
        # np.save(os.path.join(savedir, fn), clr_mask)
