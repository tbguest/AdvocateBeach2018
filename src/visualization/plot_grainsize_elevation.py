# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:47:31 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
%matplotlib qt5

def corr2(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

# for portability
homechar = "C:\\"

grid_spec = "dense_array2"

tide_range = range(14, 28)

counter = 0

dz_line01 = np.zeros((6,len(tide_range)-1))
sum_dz = np.zeros((6,24))

dmgs_line01 = np.zeros((6,len(tide_range)-1))
sum_dmgs = np.zeros((6,24))

dsort_line01 = np.zeros((6,len(tide_range)-1))
sum_dsort = np.zeros((6,24))

surv_col = 2

for ii in tide_range:

    tide = "tide" + str(ii)

    gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize", "beach_surveys", tide, grid_spec + ".npy")

    gpsfn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                          "GPS", "by_tide", tide, grid_spec + ".npy")

    wavefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                  "pressure", "wavestats", tide + ".npy")

    # wave data
    if not os.path.exists(wavefn):
        continue

    bar = np.load(wavefn).item()
    Hs = np.mean(bar["Hs"])

    # sediment data
    jnk = np.load(gsizefn).item()

    mgs = jnk['mean_grain_size']
    sort = jnk['sorting']
    skew = jnk['skewness']
    kurt = jnk['kurtosis']

    # GPS data
    if not os.path.exists(gpsfn):
        continue

    foo = np.load(gpsfn).item()

    x = foo['x'].reshape(6, 24)
    y = foo['y'].reshape(6, 24)
    z = foo['z'].reshape(6, 24)

    # for first iteration
    if counter == 0:

        last_z = z
        dz = z - last_z

        last_mgs = mgs
        last_sort = sort
        dmgs = mgs - last_mgs
        dsort = sort - last_sort

    else:
        dz = z - last_z
#        last_z = z

        dmgs = mgs - last_mgs
        dsort = sort - last_sort
#        last_mgs = mgs
#        last_sort = sort

        pcorr_dz_dmgs = corr2(dz,dmgs)
        pcorr_dz_dsort = corr2(dz,dsort)
        pcorr_dmgs_dsort = corr2(dmgs,dsort)

#        xcorr_dz_dmgs = signal.correlate2d(dz, dmgs, boundary='symm', mode='same')
#        xcorr_dz_dsort = signal.correlate2d(dz, dsort, boundary='symm', mode='same')
#        xcorr_dmgs_dsort = signal.correlate2d(dmgs, dsort, boundary='symm', mode='same')

#        xcorr_z_lastz = signal.correlate2d(z, last_z, boundary='symm', mode='same')
#        xcorr_mgs_lastmgs = signal.correlate2d(mgs, last_mgs, boundary='symm', mode='same')
#        xcorr_sort_lastsort = signal.correlate2d(sort, last_sort, boundary='symm', mode='same')

        # normalized:

        xcorr_dz_dmgs = signal.correlate2d((dz-np.mean(dz))/np.std(dz), (dmgs - np.mean(dmgs))/np.std(dmgs), boundary='symm', mode='same')/mgs.size
        xcorr_dz_dsort = signal.correlate2d((dz-np.mean(dz))/np.std(dz), (dsort-np.mean(dsort))/np.std(dsort), boundary='symm', mode='same')/mgs.size
        xcorr_dmgs_dsort = signal.correlate2d((dmgs - np.mean(dmgs))/np.std(dmgs), (dsort-np.mean(dsort))/np.std(dsort), boundary='symm', mode='same')/mgs.size

        xcorr_mgs_lastmgs = (signal.correlate2d((mgs - np.mean(mgs))/np.std(mgs), (last_mgs - np.mean(last_mgs))/np.std(last_mgs), boundary='symm', mode='same'))/mgs.size
        xcorr_z_lastz = signal.correlate2d((z-np.mean(z))/np.std(z), (last_z-np.mean(last_z))/np.std(last_z), boundary='symm', mode='same')/mgs.size
        xcorr_sort_lastsort = signal.correlate2d((sort-np.mean(sort))/np.std(sort), (last_sort-np.mean(last_sort))/np.std(last_sort), boundary='symm', mode='same')/mgs.size

        plt.figure(1)
        plt.plot(ii, Hs, 'bo')

#        plt.figure(2)
#        plt.plot(ii, pcorr_dz_dmgs, 'bo')
#        plt.plot(ii, pcorr_dz_dsort, 'ro')
#        plt.plot(ii, pcorr_dmgs_dsort, 'ko')

        # tide-tide change in z, gsize, sorting
        fig = plt.figure(2)
        ax1 = fig.add_subplot(211)
        ax1.plot(dz, dmgs, 'b.')
        ax1.set_ylabel('delta mgs [mm]')
        ax2 = fig.add_subplot(212)
        ax2.plot(dz, dsort, 'r.')
        ax2.set_xlabel('delta z [mm]')
        ax2.set_ylabel('delta sort [mm]')



        # tide-tide change in z, gsize, sorting
        fig = plt.figure(3)
        ax1 = fig.add_subplot(311)
        ax1.plot(ii, Hs, 'bo')
        ax1.set_ylabel('Hsig')
        ax2 = fig.add_subplot(312)
        ax2.plot(ii, np.sum(dz)*2, 'bo')
        ax2.set_ylabel('vol. change [m^3]')
        ax3 = fig.add_subplot(313)
        ax3.plot(ii, np.mean(mgs), 'bo')
        ax3.set_ylabel('mean grain size [mm]')
        ax3.set_xlabel('tide')
#        pos2 = ax2.plot(dz, dsort, 'r.')


        if counter > 0:
            sum_dz = sum_dz + dz
            dz_line01[:,counter-1] = sum_dz[:,surv_col]

            sum_dmgs = sum_dmgs + dmgs
            dmgs_line01[:,counter-1] = sum_dmgs[:,surv_col]

            sum_dsort = sum_dsort + dsort
            dsort_line01[:,counter-1] = sum_dsort[:,surv_col]


        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9,7), num=tide)
        pos1 = ax1.imshow(dz, cmap='bwr', vmin=-0.25, vmax=0.25)
        fig.colorbar(pos1, ax=ax1)
        pos2 = ax2.imshow(dmgs, cmap='bwr', vmin=-15, vmax=15)
        fig.colorbar(pos2, ax=ax2)
        pos3 = ax3.imshow(dsort, cmap='bwr', vmin=-15, vmax=15)
        fig.colorbar(pos3, ax=ax3)
        # ax1.set_ylabel(r'$\delta$ z')
        # ax2.set_ylabel(r'$\delta$ mgs')
        # ax3.set_ylabel(r'$\delta$ sort')
        # ax3.set_xlabel(r'$\delta$ z')
    #
        # 2d correlation dz-dgsize, dz-dsort, dgsize-dsort
    #    cmap_limit1 = np.max(np.abs(xcorr_dz_dmgs))
    #    cmap_limit2 = np.max(np.abs(xcorr_dz_dsort))
    #    cmap_limit3 = np.max(np.abs(xcorr_dmgs_dsort))

#        fig2, (ax21, ax22, ax23) = plt.subplots(nrows=3, ncols=1, figsize=(9,7))
#        pos21 = ax21.imshow(xcorr_dz_dmgs, cmap='bwr')
#        ax21.ylabel()
#        fig2.colorbar(pos21, ax=ax21)
#        pos22 = ax22.imshow(xcorr_dz_dsort, cmap='bwr')
#        fig2.colorbar(pos22, ax=ax22)
#        pos23 = ax23.imshow(xcorr_dmgs_dsort, cmap='bwr')
#        fig2.colorbar(pos23, ax=ax23)

    #    fig2, (ax21, ax22, ax23) = plt.subplots(nrows=3, ncols=1, figsize=(9,7))
    #    pos21 = ax21.imshow(xcorr_dz_dmgs, cmap='bwr')
    #    fig2.colorbar(pos21, ax=ax21)
    #    pos22 = ax22.imshow(xcorr_dz_dsort, cmap='bwr')
    #    fig2.colorbar(pos22, ax=ax22)
    #    pos23 = ax23.imshow(xcorr_dmgs_dsort, cmap='bwr')
    #    fig2.colorbar(pos23, ax=ax23)

#        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(9,7))
#        pos1 = ax1.imshow(mgs, cmap='inferno')
#        fig.colorbar(pos1, ax=ax1)
#        pos2 = ax2.imshow(last_mgs, cmap='inferno')
#        fig.colorbar(pos2, ax=ax2)
#        pos3 = ax3.imshow(sort, cmap='inferno')
#        fig.colorbar(pos3, ax=ax3)
#        pos4 = ax4.imshow(last_sort, cmap='inferno')
#        fig.colorbar(pos4, ax=ax4)

#        # 2d correlation z, mgs, sorting, from tide-tide
#        fig2, (ax21, ax22, ax23) = plt.subplots(nrows=3, ncols=1, figsize=(9,7))
#        pos21 = ax21.imshow(dz, cmap='bwr')
#        fig2.colorbar(pos21, ax=ax21)
#        pos22 = ax22.imshow(xcorr_mgs_lastmgs, cmap='bwr')
#        fig2.colorbar(pos22, ax=ax22)
#        pos23 = ax23.imshow(xcorr_sort_lastsort, cmap='bwr')
#        fig2.colorbar(pos23, ax=ax23)



        # update
        last_z = z
        last_mgs = mgs
        last_sort = sort

    counter = counter + 1

aspect = 1/3
fig4, (ax41, ax42, ax43) = plt.subplots(nrows=3, ncols=1, figsize=(9,6), num='dz,mgs,sort v. tide')
pos41 = ax41.imshow(dz_line01, cmap='bwr', vmin=-0.3, vmax=0.3, extent=[14,28,12,0])
ax41.set_aspect(aspect)
fig4.colorbar(pos41, ax=ax41)
pos42 = ax42.imshow(dmgs_line01, cmap='bwr', vmin=-20, vmax=20, extent=[14,28,12,0])
ax42.set_aspect(aspect)
fig4.colorbar(pos42, ax=ax42)
pos43 = ax43.imshow(dsort_line01, cmap='bwr', vmin=-20, vmax=20, extent=[14,28,12,0])
ax43.set_aspect(aspect)
fig4.colorbar(pos43, ax=ax43)
ax43.set_ylabel('cross-shore coordinate [m]')
ax43.set_xlabel('tide')





#    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,7))
#    pos1 = ax1.imshow(gsize['mean_grain_size'], cmap='inferno')
#    fig.colorbar(pos1, ax=ax1)
#    pos2 = ax2.imshow(gsize['sorting'], cmap='inferno')
#    fig.colorbar(pos2, ax=ax2)
