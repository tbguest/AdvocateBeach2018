# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:49:57 2019

@author: Tristan Guest

attempt #2: track one cobble at a time
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os



homechar = "C:\\"

tide = "tide19"
vidspec = "vid_1540304255"

imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "raw", \
                      "images", "fromVideo", tide, vidspec)
        
imgs = glob.glob(os.path.join(imgdir, 'img*.jpg'))

# preallocate for 1/x times the length of imgs
# time vec
# bank of locations
# bank of trajectories

# repositories of stone locations and trajectories
oldpt = []
#traj_bank = []

for file in imgs[200:1200:400]:
        
    im = plt.imread(file)    
    plt.figure(1)
    plt.clf()
    plt.imshow(im)
    
    if len(oldpt) > 0:
        plt.plot(oldpt[:,0], oldpt[:,1], 'ro')
        plt.draw()
#    plt.tight_layout()
    
#    np_oldpt = np.array(oldpt)
#    
#    # Step 1
#    if oldpt:
#        plt.plot(np_oldpt[:,0], np_oldpt[:,1], 'ro')
                
    newpt = np.array(plt.ginput(-1, timeout=0, show_clicks=True))
    
#    if not newpt:
#        continue
#    
    oldpt = newpt
    
    
    
