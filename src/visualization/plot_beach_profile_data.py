# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:47:31 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.signal import correlate
import json

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux

grid_spec = "cross_shore"

tide_range = range(14, 28)

counter = 0

# guess at cross-shore profile length
npts = 50

dz_line = np.zeros((npts,len(tide_range)-1))
z_line = np.zeros((npts,len(tide_range)-1))
sum_dz = np.zeros((npts,))

dmgs_line = np.zeros((npts,len(tide_range)-1))
mgs_line = np.zeros((npts,len(tide_range)-1))
sum_dmgs = np.zeros((npts,))

dsort_line = np.zeros((npts,len(tide_range)-1))
sort_line = np.zeros((npts,len(tide_range)-1))
sum_dsort = np.zeros((npts,))

# initialize
corrcoeffs = []
Hs = []
Tp = []
steepness = []
iribarren = []


fig01, ax01 = plt.subplots(3,1, num='profiles')

for ii in tide_range:

    tide = "tide" + str(ii)

    gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize", "beach_surveys", tide, grid_spec + ".json")

    gpsfn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                          "GPS", "by_tide", tide, grid_spec + ".json")

    wavefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                  "pressure", "wavestats", tide + ".json")

    # wave data
    if not os.path.exists(wavefn):
        continue

    # bar = np.load(wavefn, allow_pickle=True).item()
    with open(wavefn, 'r') as fpp:
        bar = json.load(fpp)
    Hs.append(np.mean(np.array(bar["Hs"])))
    Tp.append(np.mean(np.array(bar["Tp"])))
    steepness.append(np.mean(np.array(bar["steepness"])))
    iribarren.append(np.mean(np.array(bar["Iribarren"])))

    # sediment data
    # jnk = np.load(gsizefn, allow_pickle=True).item()
    with open(gsizefn, 'r') as fpp:
        jnk = json.load(fpp)

    mgs0 = np.array(jnk['mean_grain_size'])
    sort0 = np.array(jnk['sorting'])

    mgs = np.pad(mgs0, (0, npts-len(mgs0)), 'constant', constant_values=(np.nan,np.nan))
    sort = np.pad(sort0, (0, npts-len(sort0)), 'constant', constant_values=(np.nan,np.nan))

    # GPS data
    if not os.path.exists(gpsfn):
        continue

    # foo = np.load(gpsfn, allow_pickle=True).item()
    with open(gpsfn, 'r') as fpp:
        foo = json.load(fpp)

    # x = np.array(foo['x']).reshape(6, 24)
    # y = np.array(foo['y']).reshape(6, 24)
    # z = np.array(foo['z']).reshape(6, 24)
    x0 = np.array(foo['x'])
    y0 = np.array(foo['y'])
    z0 = np.array(foo['z'])

    x = np.pad(x0, (0, npts-len(x0)), 'constant', constant_values=(np.nan,np.nan))
    y = np.pad(y0, (0, npts-len(y0)), 'constant', constant_values=(np.nan,np.nan))
    z = np.pad(z0, (0, npts-len(z0)), 'constant', constant_values=(np.nan,np.nan))

    ax01[0].plot(y0, z0)
    ax01[1].plot(y0, mgs0)
    ax01[2].plot(y0, sort0)

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

        # update
        last_z = z
        last_mgs = mgs
        last_sort = sort

        sum_dz = sum_dz + dz
        dz_line[:,counter-1] = sum_dz

        sum_dmgs = sum_dmgs + dmgs
        dmgs_line[:,counter-1] = sum_dmgs

        sum_dsort = sum_dsort + dsort
        dsort_line[:,counter-1] = sum_dsort

        z_line[:,counter-1] = z
        mgs_line[:,counter-1] = mgs
        sort_line[:,counter-1] = sort

        # correlate dz with mgs(t-1), mgs, dmgs

        tmp_dz = dz
        tmp_dz = tmp_dz[~np.isnan(tmp_dz)]

        tmp_mgs = dmgs
        tmp_mgs = tmp_mgs[~np.isnan(tmp_mgs)]

        corrcoeffs.append(np.corrcoef(tmp_dz,tmp_mgs[:len(tmp_dz)])[0,1])


#     # xcorrtmp = np.correlate(tmp_dz, tmp_mgs[:len(tmp_dz)], mode='full')
#
#     a = (tmp_dz - np.mean(tmp_dz)) / (np.std(tmp_dz) * len(tmp_dz))
#     b = (tmp_mgs - np.mean(tmp_mgs)) / (np.std(tmp_mgs))
#     c = np.correlate(a, b, 'full')
#     lag = np.argmax(np.correlate(a, b, 'full'))
#
# lag = np.argmax(correlate(a_sig, b_sig))
# c_sig = np.roll(b, shift=int(np.ceil(lag)))
#
#
#     # plt.plot(xcorrtmp, '.')
#     plt.figure()
#     # plt.plot(c, '.')
#     plt.plot(c_sig, '.')


    counter = counter + 1

# truncate nans
maxreal = 0
for col in dz_line.T:
    candidate = np.count_nonzero(~np.isnan(col))
    if candidate > maxreal:
        maxreal = candidate


# # correlate dz with mgs(t-1), mgs, dmgs
#
# tmp_dz = dz_line[:,1]
# tmp_dz = tmp_dz[~np.isnan(tmp_dz)]
#
# tmp_mgs = mgs_line[:,1]
# tmp_mgs = tmp_mgs[~np.isnan(tmp_mgs)]
#
# fig, ax1 = plt.subplots(1, 1, num=ii)
# ax1.xcorr(tmp_dz, tmp_mgs[:len(tmp_dz)], usevlines=True, maxlags=10, normed=True, lw=2)
#

# dz_mgs_corr = plt.xcorr(tmp_dz, tmp_mgs[:len(tmp_dz)])

# plt.figure()
# plt.plot(dz_mgs_corr[0],dz_mgs_corr[1])



# FIGURES

plt.figure(100)
plt.imshow(z_line)

plt.figure(101)
plt.imshow(mgs_line)

plt.figure(102)
plt.imshow(sort_line)

fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(4,7), num='cumulative change')
ax20 = ax2[0].imshow(dz_line[:maxreal,:], cmap='bwr', vmin=-0.35, vmax=0.35)
fig2.colorbar(ax20, ax=ax2[0])
ax21 = ax2[1].imshow(dmgs_line[:maxreal,:], cmap='bwr')
fig2.colorbar(ax21, ax=ax2[1])
ax22 = ax2[2].imshow(dsort_line[:maxreal,:], cmap='bwr')
fig2.colorbar(ax22, ax=ax2[2])


plt.figure(num='Hs')
plt.plot(Hs[1:], corrcoeffs, '.')

plt.figure(num='Tp')
plt.plot(Tp[1:], corrcoeffs, '.')

plt.figure(num='steep')
plt.plot(steepness[1:], corrcoeffs, '.')

plt.figure(num='iribarren')
plt.plot(iribarren[1:], corrcoeffs, '.')
