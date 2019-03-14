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


homechar = "C:\\"

tide = "tide19"
position = "position2"
#vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2
colour = "yellow"

if colour is 'yellow':
    red_range = [175,255]
    grn_range = [175,255]
    blu_range = [0,150]
elif colour is 'orange':
    red_range = [150,255]
    grn_range = [0,130]
    blu_range = [0,110]
elif colour is 'blue':
    red_range = [0,115]
    grn_range = [0,125]
    blu_range = [118,255]

savedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                           "images","binaryMask", tide, position, vidspec, colour)

imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                      "images", "fromVideo", tide, position, vidspec)

imgs = sorted(glob.glob(os.path.join(imgdir, 'img*.jpg')))

# main loop
for file in imgs:

    im = plt.imread(file)
    # im = plt.imread(imgs[504])


    plt.figure(1).clf()
    plt.imshow(im)
    plt.tight_layout()
    plt.draw()
    plt.show()

    ###################
    ## testing
    r, g, b = cv.split(im)

    cv.checkRange(b, quiet=True, minVal=blu_range[0], maxVal=blu_range[1])
    is_blu = cv.inRange(b, blu_range[0], blu_range[1])
    is_grn = cv.inRange(g, grn_range[0], grn_range[1])
    is_red = cv.inRange(r, red_range[0], red_range[1])
    clr_mask = np.logical_and(is_red, is_grn)
    clr_mask = np.logical_and(clr_mask, is_blu)


    # clr_mask = np.zeros([im.shape[0], im.shape[1]])

    # for i in range(im.shape[0]):
    #     for j in range(im.shape[1]):
    #         if red_range[0] < im[i,j,0] < red_range[1] and \
    #             grn_range[0] < im[i,j,1] < grn_range[1] and \
    #             blu_range[0] < im[i,j,2] < blu_range[1]:
    #
    #             clr_mask[i,j] = 1
    #
    #         # elif clr == 'orange' and (im[i,j,0]-im[i,j,1]) > 35 and (im[i,j,0]-im[i,j,2]) > 35:
    #         #
    #         #     clr_mask[i,j] = 1

    #
    # plt.figure(1).clf()
    # plt.imshow(im)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()
    # plt.figure(2).clf()
    # plt.imshow(clr_mask)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()


    kernel = np.ones((3,3),np.uint8)
    erosion = cv.erode(clr_mask.astype(float),kernel,iterations = 2)
    dilation = cv.dilate(erosion,kernel,iterations = 4)

    # erode_dilate = cv.dilate(erosion,kernel,iterations = 1)
    close = cv.erode(dilation,kernel,iterations = 2)

    # plt.figure(1).clf()
    # plt.imshow(im)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()
    # plt.figure(2).clf()
    # plt.imshow(clr_mask)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()
    # plt.figure(3).clf()
    # plt.imshow(close)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()

    # hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    # h, s, v = cv.split(hsv)
    #########################

    fn = file[-13:-4] + '.jpg'

    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.imsave(os.path.join(savedir, fn), close)

    # cv.imwrite(os.path.join(savedir, fn), close)
    # np.save(os.path.join(savedir, fn), clr_mask)
