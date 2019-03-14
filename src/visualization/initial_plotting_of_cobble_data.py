# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 12:38:21 2019

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import errno


homechar = "C:\\"

tide = "tide19"
position = "position2"
#vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2
colour = "yellow"

cobbledir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                           "cobble_tracking", tide, position, vidspec, colour)

cobfiles = sorted(glob.glob(os.path.join(cobbledir, 'img*.npy')))

cobblecount = []
tstamp = []
xmean = []
ymean = []

for file in cobfiles:
    cobblecount.append(len(np.load(file).item()['positions']))
    tstamp.append(np.load(file).item()['timestamp'])

    x = np.array(np.load(file).item()['positions'])
    if len(x) > 0:
        xmean.append(np.mean(x[:,0]))
        ymean.append(np.mean(x[:,1]))
    else:
        xmean.append(0.0)
        ymean.append(0.0)

plt.figure(1)
plt.subplot(311)
plt.plot(tstamp, cobblecount)
plt.ylabel('no. of stones in frame')
plt.subplot(312)
plt.plot(tstamp, xmean)
plt.ylabel('x mean [pixels]')
plt.subplot(313)
plt.plot(tstamp, ymean)
plt.ylabel('y mean [pixels]')
plt.xlabel('time [s]')
