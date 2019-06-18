#!/usr/bin/env python3
'''
Tristan B Guest
8 Mar 2019
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import cv2 as cv
from PIL import Image


# plt.close("all")

%matplotlib qt5

# homechar = "C:\\"
homechar = os.path.expanduser('~')

tide = "tide19"
positions = ["position1","position2","position3","position4"]

for position in positions:

    plt.close("all")

    if position == "position1":
        vidspec = "vid_1540304255" # pos1
        imgnum = 'img010413.jpg' #1
    elif position == "position2":
        vidspec = "vid_1540307860" # pos2
        imgnum = 'img010827.jpg' #2
    elif position == "position3":
        vidspec = "vid_1540307860" # pos3
        imgnum = 'img013983.jpg' #3
    else:
        vidspec = "vid_1540311466" # pos4
        imgnum = 'img003238.jpg' #4


    cmap = cm.tab10

    # imgfile = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
    #                            "images", "fromVideo", tide,position,vidspec,imgnum)
    imgfile = os.path.join('/media', 'tristan2','Advocate2018_backup2', "data", "interim", \
                               "images", "fromVideo", tide,position,vidspec,imgnum)
    im = plt.imread(imgfile)

    posixtime0 = float(vidspec[-10:])

    fig, ax = plt.subplots(nrows=1,ncols=1, num="trajectory overlay")
    ax.imshow(im)
    # ax.plot(np.arange(len(im[500,:])), im[500,:, 1])

    rescale_pix_x = 1640/1000
    rescale_pix_y = 1232/1000

    edges = cv.Canny(im, 100, 200)/255

    fig2, ax2 = plt.subplots(1,1,num='edges')
    ax2.imshow(edges)


    # divide the image into nxm subregions
    n = 20
    m = n

    xlen = len(im[0,:,0])
    ylen = len(im[:,0,0])

    xstep = xlen/n
    ystep = ylen/m

    # omit highest cell, so can be added in loop
    xgrid = np.linspace(0,xlen-xstep,np.floor(np.floor(xlen/xstep))).astype(int)
    ygrid = np.linspace(0,ylen-ystep,np.floor(np.floor(ylen/ystep))).astype(int)

    coarsefine_split = []

    # for coarse-fine split
    split_image = np.zeros(np.shape(im[:,:,1]))

    for I_x in xgrid:
        for  I_y in ygrid:
            edgecount = np.sum(edges[I_y:I_y+int(np.floor(ystep)),I_x:I_x+int(np.floor(xstep))])
            # print(edgecount)
            if edgecount/len(edges[I_y:I_y+int(np.floor(ystep)),I_x:I_x+int(np.floor(xstep))].flatten()) > 0.13:
                coarsefine_split.append(1)
                split_image[I_y:I_y+int(np.floor(ystep)),I_x:I_x+int(np.floor(xstep))] = 1
            else:
                coarsefine_split.append(0)

    coarsefine_split = np.array(coarsefine_split).reshape(len(ygrid),len(xgrid))

    fig3, ax3 = plt.subplots(1,1,num='split')
    # ax3.imshow(coarsefine_split.T)
    ax3.imshow(split_image)

    fig4, ax4 = plt.subplots(nrows=1,ncols=1, num="original image")
    ax4.imshow(im)

    ax.imshow(split_image, alpha=0.4)


    saveFlag = 1
    if saveFlag == 1:

        savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','cobble_transport','coarsefine_split',tide)
        if not os.path.exists(savedn):
            try:
                os.makedirs(savedn)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        fig.savefig(os.path.join(savedn, 'overlay_split-' + position + '.png'), dpi=1000, transparent=True)
        fig.savefig(os.path.join(savedn, 'overlay_split-' + position + '.pdf'), dpi=None, transparent=True)

        fig2.savefig(os.path.join(savedn, 'edges-' + position + '.png'), dpi=1000, transparent=True)
        fig2.savefig(os.path.join(savedn, 'edges-' + position + '.pdf'), dpi=None, transparent=True)

        fig3.savefig(os.path.join(savedn, 'binary-' + position + '.png'), dpi=1000, transparent=True)
        fig3.savefig(os.path.join(savedn, 'binary-' + position + '.pdf'), dpi=None, transparent=True)

        fig4.savefig(os.path.join(savedn, 'original_image-' + position + '.png'), dpi=1000, transparent=True)
        fig4.savefig(os.path.join(savedn, 'original_image-' + position + '.pdf'), dpi=None, transparent=True)
