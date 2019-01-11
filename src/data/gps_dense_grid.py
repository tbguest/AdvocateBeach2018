# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:21:43 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def gsize_stats(fname):

    import csv

    with open(fname, 'rt') as f:

        readCSV = csv.reader(f, delimiter=',')
        skewness = []
        mean_gsize = []
        percentiles = []
        percentile_vals = []
        gsize_freqs = []
        gsize_bins = []
        sorting = []
        kurtosis = []

        for col in readCSV:

            skewness.append(col[0])
            mean_gsize.append(col[1])
            percentiles.append(col[2])
            percentile_vals.append(col[3])
            gsize_freqs.append(col[4])
            gsize_bins.append(col[5])
            sorting.append(col[6])
            kurtosis.append(col[7])
            
        return skewness, mean_gsize, percentiles, percentile_vals, gsize_freqs, gsize_bins, sorting, kurtosis
    
    
def reformat_gsize_stats(day,tide,grid_specs):

    import glob

    dn = "C:\\Projects\\AdvocateBeach2018\\data\\processed\\grainsize_dists\\" \
            + day + "\\" + tide + "\\" + grid_specs + "\\"

    skew_vec = []
    mean_gsize_vec = []
    sort_vec = []
    kurt_vec = []

    for fname in glob.glob(dn + '*.csv'):

        skewness, mean_gsize, percentiles, percentile_vals, gsize_freqs, gsize_bins, sorting, kurtosis = gsize_stats(fname)

        skew_vec.append(float(skewness[1]))
        mean_gsize_vec.append(float(mean_gsize[1]))
        sort_vec.append(float(sorting[1]))
        kurt_vec.append(float(kurtosis[1]))
        
    return mean_gsize_vec, sort_vec, skew_vec, kurt_vec


# tide number - 1 being first of experiment (Sunday AM UTC)
#{"15_10_2018_A.txt": 3, "16_10_2018_A.txt": 5, \
tide_table = {"17_10_2018_A": 7, "18_10_2018_A": 9, "19_10_2018_A": 11, "20_10_2018_A": 13, \
    "21_10_2018_A": 14, "21_10_2018_B": 15, "22_10_2018_A": 16, "22_10_2018_B": 17, "23_10_2018_A": 18, \
    "23_10_2018_B": 19, "24_10_2018_A": 20, "24_10_2018_B": 21, "25_10_2018_A": 22, \
    "25_10_2018_B": 23, "26_10_2018_A": 24, "26_10_2018_B": 25, "27_10_2018_A": 26, \
    "27_10_2018_B": 27}
  
# for portability
homechar = "C:\\"

# grain size
days = ["21_10_2018", "21_10_2018", "22_10_2018", "22_10_2018", "23_10_2018", \
        "23_10_2018", "24_10_2018", "24_10_2018", "25_10_2018"]
tides = ["AM","PM","AM","PM","AM","PM","AM","PM","AM"]
grid_spec = "dense_array2"

gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                  "gsize_dists")

day = days[0]
tide = tides[0]
dn = "C:\\Projects\\AdvocateBeach2018\\data\\processed\\grainsize_dists\\" + \
        day + "\\" + tide + "\\" + grid_spec + "\\"




dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                  "GPS", "by_tide")

wavesdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                  "pressure", "wavestats")

#dlist = os.listdir(dn)
tideindex = range(1, 28)

counter = 0

dvol = []
good_GPS = []

good_waves = []
Hs = []
Tp = []
steepness = []
iribarren = []

#for folder in dlist:
for ii in tideindex:
    
    # wave data
    if not os.path.exists(os.path.join(wavesdir, "tide" + str(ii)) + ".npy"):
        continue
    
    jnk = np.load(os.path.join(wavesdir, "tide" + str(ii)) + ".npy")
    jnk2 = jnk[()]
    Hs.append(np.mean(jnk2["Hs"]))
    Tp.append(np.mean(jnk2["Tp"]))
    steepness.append(np.mean(jnk2["steepness"]))
    iribarren.append(np.mean(jnk2["Iribarren"]))
    good_waves.append(ii)

    # GPS data
    if not os.path.isdir(os.path.join(dn, "tide" + str(ii))):
        continue
    
    xx = np.load(os.path.join(dn, "tide" + str(ii), "dense_array2.npy"))
#    xx = np.load(os.path.join(dn, "tide" + str(ii), "longshore2.npy"))
    
#    xx = np.load(os.path.join(dn, folder, "dense_array2.npy"))
    yy = xx[()] # this seems to be necessary to access data
    
    x = yy['x']
    y = yy['y']
    z = yy['z']
    
    if len(z) == 0:
        continue
    
    # for first iteration
    if counter == 0:
        last_z = z
        dz = z - last_z
    else:
        dz = z - last_z
        last_z = z
     
    dvol.append(np.sum(dz) * 2)  
    good_GPS.append(ii)
    
    counter = counter + 1    
        
#    if counter == 10:     
    
#    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))
#    pos = ax.imshow(dz.reshape(6, 24), cmap='bwr')        
#    fig.colorbar(pos, ax=ax)

    
plt.figure(1)
plt.subplot(511),
plt.plot(good_GPS, dvol)
plt.xlim(10, 28)
plt.subplot(512)
plt.plot(good_waves, Hs)
plt.xlim(10, 28)
plt.subplot(513)
plt.plot(good_waves, Tp)
plt.xlim(10, 28)
plt.subplot(514)
plt.plot(good_waves, steepness)
plt.xlim(10, 28)
plt.subplot(515)
plt.plot(good_waves, iribarren)
plt.xlim(10, 28)


