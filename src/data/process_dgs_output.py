# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:00 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def build_gsize_arrays(gsizedir):
    
    allfn = os.listdir(gsizedir)

    mean_gsize = []
    sort = []
    skew = []
    kurt = []
    
    for fn in allfn:
        
        jnk = np.load(os.path.join(gsizedir, fn), encoding='latin1').item() # unpickle python 2 -> 3
        
        mean_gsize.append(jnk['mean grain size'])
        sort.append(jnk['grain size sorting'])
        skew.append(jnk['grain size skewness'])
        kurt.append(jnk['grain size kurtosis'])
        
    mean_gsize = np.array(mean_gsize).reshape(6, 24)
    sort = np.array(sort).reshape(6, 24)
    skew = np.array(mean_gsize).reshape(6, 24)
    kurt = np.array(sort).reshape(6, 24)
    
    gsize = {"mean_grain_size": mean_gsize, \
             "sorting": sort, \
             "skewness": skew, \
             "kurtosis": kurt}
    
    return gsize


def main():

    tide_range = range(14, 28)

    ## load key:value dict to convert from yearday to tide num.
    #tidekeydn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "external", "tide_key_values.npy")
    #tidekey = np.load(tidekeydn).item()
    
    # for portability
    homechar = "C:\\"
    
    grid_spec = "dense_array2"
    
    for ii in tide_range:
    
        tide = "tide" + str(ii)
        
        gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize_dists", tide, grid_spec)
        
        outdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize_dists", tide, grid_spec + ".npy")
        
        gsize = build_gsize_arrays(gsizedir)
    
        np.save(outdir, gsize)
    
#    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,7))
#    pos1 = ax1.imshow(gsize['mean_grain_size'], cmap='inferno')        
#    fig.colorbar(pos1, ax=ax1)  
#    pos2 = ax2.imshow(gsize['sorting'], cmap='inferno')        
#    fig.colorbar(pos2, ax=ax2)   
        
        
main()        
