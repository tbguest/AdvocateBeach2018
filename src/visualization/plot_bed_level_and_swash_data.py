# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:45:01 2019

@author: Owner

plot range and grain size data, by chunk

array({'19_10_2018_A': 11, '27_10_2018_B': 27, '23_10_2018_B': 19, '22_10_2018_B': 17, '23_10_2018_A': 18, '17_10_2018_A': 7, '21_10_2018_B': 15, '24_10_2018_B': 21, '27_10_2018_A': 26, '25_10_2018_A': 22, '24_10_2018_A': 20, '20_10_2018_A': 13, '26_10_2018_B': 25, '21_10_2018_A': 14, '16_10_2018_A': 5, '18_10_2018_A': 9, '26_10_2018_A': 24, '15_10_2018_A': 3, '25_10_2018_B': 23, '22_10_2018_A': 16},
      dtype=object)

"""

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import sys
# from .data.regresstools import lowess
from src.data.regresstools import lowess
from src.data.regresstools import loess_1d
from datetime import datetime
import time
import matplotlib.dates as md
import matplotlib.mlab as mlab
from scipy.stats import norm

# %matplotlib inline

# change default font size
plt.rcParams.update({'font.size': 12})


def save_figures(dn, fn, fig):
    ''' Saves png and pdf of figure.

    INPUTS
    dn: save directory. will be created if doesn't exist
    fn: file name WITHOUT extension
    fig: figure handle
    '''

    dn0 = os.path.join(dn, 'png')
    dn1 = os.path.join(dn, 'pdf')
    dn2 = os.path.join(dn, 'eps')
    dn3 = os.path.join(dn, 'jpg')

    if not os.path.exists(dn0):
        os.makedirs(dn0)
    if not os.path.exists(dn1):
        os.makedirs(dn1)
    if not os.path.exists(dn2):
        os.makedirs(dn2)
    if not os.path.exists(dn3):
        os.makedirs(dn3)

    fig.savefig(os.path.join(dn0, fn + '.png'), dpi=1000, transparent=True)
    fig.savefig(os.path.join(dn1, fn + '.pdf'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn2, fn + '.eps'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn3, fn + '.jpg'), dpi=1000, transparent=True)


def utime2yearday(unixtime):
    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday


def loess_fit(time_data, grain_size_data):

    f0 = 0.125
    iters = 3
    smoothed_data = lowess.lowess(time_data, grain_size_data, f=f0, iter=iters)

    return smoothed_data

# tide = '15'
tide = '19'
# tide = '27'

chunk = '2'

## I tried looping over chunks, so more points coupld be plotted on histogram, scatterplots
## The result wasn't much better, and makes the code a mess...
# chunks = ['1','2','3','4','5']
# chunks = ['1','2','3','4']
# chunks = ['1','2','3']
# delta_bed_histogram = []
# fig02, ax02 = plt.subplots(1,1, figsize=(5,7), num='swash depth vs $\Delta$ bed level')
# for chunk in chunks:


# for saving plots
saveFlag = 0

post = 1.7 # survey post height

# homechar = "C:\\"
homechar = os.path.expanduser('~')

gsdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'grainsize', 'pi_array', 'tide' + tide)

beddir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'bed_level', 'tide' + tide)

swshdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'swash', 'tide' + tide)

# pi locations
pifile = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'GPS', 'array', 'tide' + tide, \
                           'array_position' + chunk + '.npy')


gs_struct = np.load(os.path.join(gsdir, 'chunk' + chunk + '.npy'), allow_pickle=True).item()
bedlevel_struct = np.load(os.path.join(beddir, 'chunk' + chunk + '.npy'), allow_pickle=True).item()
swash_struct = np.load(os.path.join(swshdir, 'chunk' + chunk + '.npy'), allow_pickle=True).item()

pilocs = np.load(pifile, allow_pickle=True).item()

pinums = ['pi71', 'pi72', 'pi73', 'pi74']

# initialize PLOTS
fig01, ax01 = plt.subplots(4,1, figsize=(7,9), num='MSD timeseries chunk '+chunk)
fig02, ax02 = plt.subplots(1,1, figsize=(5,7), num='swash depth vs $\Delta$ bed level. chunk '+chunk)

counter = -1
piloc_counter = -1
legend_xcoords = []

delta_bed_histogram = []
handles01 = []

# plot params
xfmt = md.DateFormatter('%H:%M')
gridalpha = 0.4

# to be iterated eventually
for pinum in pinums:

    # pinum = pinums[0]

    if tide != '27':
        counter += 1

    piloc_counter += 1 # since I don't always increase the other counter


    if not pinum in gs_struct.keys():
        continue

    piloc_x_str = str(pilocs['x'][piloc_counter])
    piloc_x = piloc_x_str[:piloc_x_str.index('.')+2]
    legend_xcoords.append('x='+piloc_x+' m')

    z_offset = pilocs['z'][counter] + post

    # grain size data defs
    t_gs = gs_struct[pinum]['tvec'] - 3*60*60
    mgs = gs_struct[pinum]['mgs']
    sort = gs_struct[pinum]['sort']
    date_gs = [datetime.fromtimestamp(x) for x in t_gs]

    # smooth grain size data
    mgs_fit = loess_fit(t_gs, mgs)
    sort_fit = loess_fit(t_gs, sort)

    # bed level definitions
    z_bed = z_offset - bedlevel_struct['data']['bed'][pinum]
    t_z_bed = bedlevel_struct['data']['time_bed'][pinum] - 3*60*60
    date_z_bed = [datetime.fromtimestamp(x) for x in t_z_bed]


    # swash definitions

    # full swash
    swash = swash_struct['data']['swash'][pinum]
    t_swash = swash_struct['data']['time_swash'][pinum] - 3*60*60
    swash_plus_bed = z_offset - swash_struct['data']['swash_plus_bed'][pinum]
    date_swash = [datetime.fromtimestamp(x) for x in t_swash]

    # peaks only
    swash_peaks = swash_struct['data']['swash_peaks'][pinum]
    t_swash_peaks = swash_struct['data']['time_swash_peaks'][pinum] - 3*60*60
    swash_peaks_plus_bed = z_offset - swash_struct['data']['swash_peaks_plus_bed'][pinum]
    date_swash_peaks = [datetime.fromtimestamp(x) for x in t_swash_peaks]

    # swash peaks in vicinity of delta(bed level)
    swash_depth = swash_struct['data']['swash_depth'][pinum]
    delta_bed = swash_struct['data']['delta_bed'][pinum]*1000
    delta_bed_histogram.extend(delta_bed) # for plotting histogram that includes all sensors

    # plot limits
    tmin = t_z_bed[0]
    tmax = t_z_bed[-1]
    datemin = date_z_bed[0]
    datemax = date_z_bed[-1]

    # pi-specific plotting colors
    if pinum == 'pi71':
        clr = 'C0'
        zorder = 15
    elif pinum == 'pi72':
        clr = 'C1'
        zorder = 10
    elif pinum == 'pi73':
        clr = 'C2'
        # clr = 'C4'
        zorder = 5
    else:
        clr = 'C3'
        zorder = 0

    # plot MSD time series
    ax01[0].plot(date_swash, swash, '-', color=clr, Linewidth=0.25, zorder=zorder)
    # ax01[0].plot(date_swash, swash, '-', color=clr, Linewidth=0.25)

    ax01[1].plot(date_z_bed, z_bed, '.', color=clr, Markersize=3)

    ax01[2].plot(date_gs, mgs, '.', color=clr, Markersize=3)
    tmp_hndl = ax01[2].plot(date_gs, mgs_fit, '-', color=clr, Linewidth=2)
    handles01.extend(tmp_hndl)

    ax01[3].plot(date_gs, sort, '.', color=clr, Markersize=3)
    ax01[3].plot(date_gs, sort_fit, '-', color=clr, Linewidth=2)


    # scatter plot
    ax02.plot(swash_depth*1000, np.abs(delta_bed),'.')




    # FIGURES TO BE SAVED IN LOOP:

    fig101 = plt.figure(num=pinum + ' swash with marked peaks and bed level')
    ax101 = fig101.add_subplot(211)
    # ax21.plot(newtt, cut_noise)
    # ax21.plot(date_good,bed_good, '.')
    # ax21.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
    ax101.plot(date_swash, swash_plus_bed, Linewidth=0.25)
    ax101.plot(date_z_bed, z_bed, '.', Markersize=5)
    ax101.plot(date_swash_peaks, swash_peaks_plus_bed,'.', Markersize=5)
    ax101.set_xlabel('time [UTC]')
    ax101.set_ylabel('z [m]')
    ax101.autoscale(enable=True, axis='x', tight=True)
    ax101.xaxis.set_major_formatter(xfmt)
    # ax21.invert_yaxis()
    # xfmt = md.DateFormatter('%H:%M')
    # ax21.xaxis.set_major_formatter(xfmt)

    ax102 = fig101.add_subplot(212)
    # ax22.plot(newtt, cut_noise)
    # ax22.plot(date_good,bed_good, '.')
    # ax22.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
    ax102.plot(date_swash, swash_plus_bed)
    ax102.plot(date_z_bed, z_bed, '.')
    ax102.plot(date_swash_peaks, swash_peaks_plus_bed,'.')
    ax102.set_xlabel('time [UTC]')
    ax102.set_ylabel('z [m]')
    # ax22.set_xlim([datetime.fromtimestamp(np.min(t_swash)), \
    #     datetime.fromtimestamp(np.min(t_swash) + 1/10*(np.max(t_swash) - np.min(t_swash)))])
    ax102.set_xlim([np.min(date_swash), np.min(date_swash) + 1/10*(np.max(date_swash) - np.min(date_swash))])
    # ax22.autoscale(enable=True, axis='x', tight=True)
    ax102.autoscale(enable=True, axis='y', tight=True)
    ax102.xaxis.set_major_formatter(xfmt)
    # ax22.invert_yaxis()
    # ax22.xaxis.set_major_formatter(xfmt)
    fig101.tight_layout()


    fig103 = plt.figure(num=pinum + '$\Delta z$ histogram')
    plt.hist(delta_bed*1000, bins=20)
    plt.xlabel('bed level change [mm]')
    plt.ylabel('occurrences')


    fig301 = plt.figure(num=pinum + '; swash; swash peaks')
    ax301 = fig301.add_subplot(211)
    ax301.plot(date_swash, swash*1000)
    ax301.plot(date_swash_peaks, swash_peaks*1000, '.')
    ax301.set_xlabel('time [UTC]')
    ax301.set_ylabel('$h_{s}$ [mm]')
    ax301.xaxis.set_major_formatter(xfmt)

    ax302 = fig301.add_subplot(212)
    ax302.plot(date_swash, swash*1000)
    ax302.plot(date_swash_peaks, swash_peaks*1000, '.')
    ax302.set_xlabel('time [UTC]')
    ax302.set_ylabel('$h_{s}$ [mm]')
    ax302.set_xlim([np.min(date_swash), np.min(date_swash) + 1/10*(np.max(date_swash) - np.min(date_swash))])
    # ax22.autoscale(enable=True, axis='x', tight=True)
    ax302.autoscale(enable=True, axis='y', tight=True)
    ax302.xaxis.set_major_formatter(xfmt)
    fig301.tight_layout()



    # export figs
    if saveFlag == 1:
        loopdn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD','tide'+tide, 'chunk'+chunk)

        save_figures(loopdn, pinum + '_swash_peaks_bed_level', fig101)
        save_figures(loopdn, pinum + '_swash_peaks_no_bed_level', fig301)








# FIGURE 01

ax01[0].set_xlim([datemin, datemax])
ax01[0].xaxis.set_major_formatter(xfmt)
ax01[0].xaxis.set_major_formatter(plt.NullFormatter())
ax01[0].grid(alpha=gridalpha)
ax01[0].tick_params(direction='in')
ax01[0].legend(handles01, legend_xcoords, loc="upper right")
# ax01[0].autoscale(enable=True, axis='y', tight=True)
# ax01[0].set_ylim([0, __])
ax01[0].set_ylabel('$h_{s}$ [m]')
ax01[1].set_xlim([datemin, datemax])
ax01[1].xaxis.set_major_formatter(xfmt)
ax01[1].xaxis.set_major_formatter(plt.NullFormatter())
ax01[1].grid(alpha=gridalpha)
ax01[1].tick_params(direction='in')
ax01[1].set_ylabel('$z$ [m]')
ax01[2].set_xlim([datemin, datemax])
ax01[2].xaxis.set_major_formatter(xfmt)
ax01[2].xaxis.set_major_formatter(plt.NullFormatter())
ax01[2].grid(alpha=gridalpha)
ax01[2].tick_params(direction='in')
ax01[2].set_ylabel('$M_0$ [mm]')
ax01[3].set_xlim([datemin, datemax])
ax01[3].xaxis.set_major_formatter(xfmt)
ax01[3].grid(alpha=gridalpha)
ax01[3].tick_params(direction='in')
ax01[3].set_ylabel('$M_1$ [mm]')
ax01[3].set_xlabel('time [UTC]')
fig01.tight_layout()


# FIGURE 02

ax02.set_xlabel('swash depth [mm]')
ax02.set_ylabel('absolute bed level change [mm]')
fig02.tight_layout()

# FIGURE 03

delta_bed_histogram = np.squeeze(delta_bed_histogram)
delta_bed_histogram = delta_bed_histogram[np.abs(delta_bed_histogram) < 30]

# best fit of data
(mu, sigma) = norm.fit(delta_bed_histogram)

# the histogram of the data
fig03 = plt.figure(num='$\Delta z$ histogram')
n, bins, patches = plt.hist(delta_bed_histogram, 30, density=1, alpha=0.75)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

# fig03 = plt.figure(num='$\Delta z$ histogram')
# plt.hist(delta_bed_histogram*1000, bins=30)
plt.xlabel('bed level change [mm]')
plt.ylabel('occurrences')



# export figs
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD','tide'+tide, 'chunk'+chunk)

    save_figures(savedn, 'MSD_timeseries', fig01)
    save_figures(savedn, 'delta_bed_level_swash_depth_scatter', fig02)
    save_figures(savedn, 'delta_bed_level_histogram', fig03)
