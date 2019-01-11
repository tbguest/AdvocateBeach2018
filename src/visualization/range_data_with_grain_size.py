# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:34:20 2019

@author: Tristan Guest

This code is destined for the chopping block -- will be superceded by consoidate_digital_grain_size_output
"""

import numpy as np 
import os
from glob import glob
import matplotlib.pyplot as plt


tide = '19'
pinum = '71'

homechar = "C:\\"
rangedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
            "range_data", "bed_level", "tide" + tide)

gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
            "grainsize_dists", "pi_array", "tide" + tide, 'pi' + pinum)
#gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
#            "grainsize_dists", "pi_array", "tide" + tide, 'pi' + pinum, "smooth_bed_level")

#C:\Projects\AdvocateBeach2018\data\interim\grainsize_dists\pi_array\tide19\pi71

#for file in glob(os.path.join(rangedir, 'sonar*.npy')):
for file in glob(os.path.join(rangedir, 'sonar' + pinum + '.npy')):
    jnk = np.load(file).item()

rng = jnk['raw bed level'][1]
tt = jnk['raw bed level'][0]

mean_gs = []
std_gs = []
timg = []

for file in glob(os.path.join(gsizedir, 'img*.npy')):
    
    foo = np.load(file, encoding='latin1').item()
    timg.append(float(file[-29:-19] + '.' + file[-18:-12]))
    mean_gs.append(foo['mean grain size'])
    std_gs.append(foo['grain size sorting'])
    
tt_img = np.array(timg) + 3.0*60*60
    
xmin = np.min(tt_img)    
xmax = np.max(tt_img)    
#
#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8,6))
#ax1.plot(tt, rng, '.') 
#ax1.set_ylabel('range [mm]')
##ax1.set_xlim(xmin, xmax)
#ax2.plot(tt_img, mean_gs, '.')   #+ 3.0*60*60
##ax2.set_xlabel('unix time [s]')
#ax2.set_ylabel('mgs [mm]')
##ax2.set_xlim(xmin, xmax)
#ax3.plot(tt_img, std_gs, '.')   #+ 3.0*60*60
#ax3.set_xlabel('unix time [s]')
#ax3.set_ylabel('mgs [mm]')

plt.figure(101)
plt.subplot(311)
plt.plot(tt, rng, '.') 
plt.ylabel('range [mm]')
#ax1.set_xlim(xmin, xmax)
plt.subplot(312)
plt.plot(tt_img, mean_gs, '.')   #+ 3.0*60*60
#ax2.set_xlabel('unix time [s]')
plt.ylabel('mgs [mm]')
#ax2.set_xlim(xmin, xmax)
plt.subplot(313)
plt.plot(tt_img, std_gs, '.')   #+ 3.0*60*60
plt.xlabel('unix time [s]')
plt.ylabel('mgs [mm]')
    
 

