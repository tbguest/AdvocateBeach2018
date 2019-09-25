
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal
import time
from datetime import datetime
# %matplotlib qt5



def rotate_coordainates(nrth, east, elvn):

    ### TEMP:
    # fname = posfiles[0]
    ###

    # UTM coords of origin (Adv2015)
    originx = 3.579093296000000e+05;
    originy = 5.022719408400000e+06;

    northing = nrth - originx
    easting = east - originy
    z = elvn

    # rotation angle from 2015 stake coordinates
    theta_rot = np.pi + np.arctan((357899.3979000000 - 357908.4006000000)/(5022717.801400000 - 5022709.436700001))

    # rotation matrix
    R = np.array([np.cos(theta_rot), -np.sin(theta_rot), np.sin(theta_rot), np.cos(theta_rot)]).reshape((2, 2))

    # apply to coordinates
    Ymat = np.array([northing, easting])
    Ymat_p = np.matmul(R,Ymat)
    y = Ymat_p[0]
    x = Ymat_p[1]

    return x, y, z


homechar = os.path.expanduser('~')

tide = "tide19"

imgs = ['img005100.jpg' , 'img011860.jpg', 'img000097.jpg', 'img005757.jpg']


for ii in range(1, 5):
    dn = '/media/tristan2/Advocate2018_backup2/data/interim/images/fromVideo/tide19/position' + str(ii) + '/targets'

    imgfile = os.path.join(dn, imgs[ii-1])

    im = plt.imread(imgfile)

    plt.figure(ii)
    plt.imshow(im)


# 1
inv_pix = 1640 - 498 # ctarg cross-shore pix coord
CTARG = [357932.8343,5022713.9053,4.1869]
scaling = 421.0550841144531 # pix/m
x, y, z = rotate_coordainates(CTARG[0], CTARG[1], CTARG[2])
# cross-shore corrd of target is y;
# target is 498 pixels from seaward edge of image
y
y - inv_pix/scaling
y + 498/scaling



# 2
inv_pix = 1640 - 859
CTARG = [357930.5826,5022710.5904,3.5734]
x, y, z = rotate_coordainates(CTARG[0], CTARG[1], CTARG[2])
scaling = 458.05166513842414
y
y - inv_pix/scaling
y + 859/scaling



# 3
inv_pix = 1640 - 1067
CTARG = [357928.2278,5022707.4896,3.1164]
x, y, z = rotate_coordainates(CTARG[0], CTARG[1], CTARG[2])
scaling = 472.43762017669604
y
y - inv_pix/scaling
y + 1067/scaling



# 4
inv_pix = 1640 - 658
CTARG = [357924.0909,5022705.8421,2.6544]
x, y, z = rotate_coordainates(CTARG[0], CTARG[1], CTARG[2])
scaling = 436.65137206616646
y
y - inv_pix/scaling
y + 658/scaling
