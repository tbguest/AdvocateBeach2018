# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:45:01 2019

@author: Owner

plot range and grain size data, by chunk

array({'19_10_2018_A': 11, '27_10_2018_B': 27, '23_10_2018_B': 19, '22_10_2018_B': 17, '23_10_2018_A': 18, '17_10_2018_A': 7, '21_10_2018_B': 15, '24_10_2018_B': 21, '27_10_2018_A': 26, '25_10_2018_A': 22, '24_10_2018_A': 20, '20_10_2018_A': 13, '26_10_2018_B': 25, '21_10_2018_A': 14, '16_10_2018_A': 5, '18_10_2018_A': 9, '26_10_2018_A': 24, '15_10_2018_A': 3, '25_10_2018_B': 23, '22_10_2018_A': 16},
      dtype=object)

"""

%reset -f


import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import sys
# from .data.regresstools import lowess
from src.data.regresstools import lowess
from src.data.regresstools import loess_1d
from datetime import datetime
from datetime import timedelta
import time
import matplotlib.dates as md
import matplotlib.mlab as mlab
from scipy.stats import norm

# %matplotlib inline
# %matplotlib qt5

# change default font size
plt.rcParams.update({'font.size': 12})

plt.close("all")



def get_last_bed_levels(bedlevel_clumps, t_bedlevel_nonans):
    ''' picks last value (and time) for each bed signal chunk '''

    last_bedlevel = []
    time_last_bedlevel = []

    time_tally = 0
    for clmp in bedlevel_clumps:
        clump_len = len(clmp)

        # in case I want to include a duration threshold (if so, set as > 1)
        if clump_len < 1:
            time_tally += clump_len
            continue

        clmp_time = t_bedlevel_nonans[time_tally:time_tally + clump_len]

        last_bedlevel.append(clmp[-1])
        time_last_bedlevel.append(clmp_time[-1])

        time_tally += clump_len

    return np.squeeze(last_bedlevel), np.squeeze(time_last_bedlevel)


def using_clump(a):
    '''takes a 1d array of values containing nans, and separates into list of
    arrays demarcated by nans
    '''
    clumps = [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]

    return clumps

def save_figures(dn, fn, fig):
    ''' Saves png and pdf of figure.

    INPUTS
    dn: save directory. will be created if doesn't exist
    fn: file name WITHOUT extension
    fig: figure handle
    '''

    # dn0 = os.path.join(dn, 'png')
    # dn1 = os.path.join(dn, 'pdf')
    # dn2 = os.path.join(dn, 'eps')
    # dn3 = os.path.join(dn, 'jpg')

    if not os.path.exists(dn):
        os.makedirs(dn)
    # if not os.path.exists(dn1):
    #     os.makedirs(dn1)
    # if not os.path.exists(dn2):
    #     os.makedirs(dn2)
    # if not os.path.exists(dn3):
    #     os.makedirs(dn3)

    fig.savefig(os.path.join(dn, fn + '.png'), dpi=1000, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.pdf'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.eps'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.jpg'), dpi=1000, transparent=True)


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
# tide = '19'
tide = '27'

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
# homechar = os.path.expanduser('~')
# homechar = "F:\\NovBackup"
homechar = os.path.join('/media','tristan2','Advocate2018_backup2')


# gsdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
#                            'processed', 'grainsize', 'pi_array', 'tide' + tide)
#
# beddir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
#                            'processed', 'range_data', 'bed_level', 'tide' + tide)
#
# swshdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
#                            'processed', 'range_data', 'swash', 'tide' + tide)
#
# # pi locations
# pifile = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
#                            'processed', 'GPS', 'array', 'tide' + tide, \
#                            'array_position' + chunk + '.npy')

# gsdir = os.path.join(homechar, 'processed', 'grainsize', 'pi_array', 'tide' + tide)
gsdir = os.path.join(homechar, 'data','processed', 'grainsize', 'pi_array', 'tide' + tide, 'reprocessed_x10')

beddir = os.path.join(homechar, 'data', 'processed', 'range_data', 'bed_level', 'tide' + tide)

swshdir = os.path.join(homechar, 'data', 'processed', 'range_data', 'swash', 'tide' + tide)

# pi locations
pifile = os.path.join(homechar, 'data', 'processed', 'GPS', 'array', 'tide' + tide, \
                           'array_position' + chunk + '.npy')



gs_struct = np.load(os.path.join(gsdir, 'chunk' + chunk + '.npy'), allow_pickle=True).item()
bedlevel_struct = np.load(os.path.join(beddir, 'chunk' + chunk + '.npy'), allow_pickle=True).item()
swash_struct = np.load(os.path.join(swshdir, 'chunk' + chunk + '.npy'), allow_pickle=True).item()

pilocs = np.load(pifile, allow_pickle=True).item()

pinums = ['pi71', 'pi72', 'pi73', 'pi74']

# initialize PLOTS
fig01, ax01 = plt.subplots(3,1, figsize=(7,6), num='MSD timeseries chunk '+chunk)
fig001, ax001 = plt.subplots(3,1, figsize=(6,4), num='MSD 1 timeseries chunk '+chunk)
fig02, ax02 = plt.subplots(1,2, figsize=(8,3.5), num='swash depth vs $\Delta$ bed level. chunk '+chunk)

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

    t_dz_bed = swash_struct['data']['time_swash_depth'][pinum] - 3*60*60
    date_dz_bed = [datetime.fromtimestamp(x) for x in t_dz_bed]

    # bed level demarcated by nans
    bedlevel = bedlevel_struct['data']['bed_level'][pinum]
    t_bedlevel = bedlevel_struct['data']['date_bed_level'][pinum] - 3*60*60



    # indices of non nan values
    I = np.argwhere(~np.isnan(bedlevel))
    # apply to time vector
    t_bedlevel_nonans = t_bedlevel[I]


    # last value and time of each bed level chunk (between instances of runup)
    bedlevel_clumps = using_clump(bedlevel)
    last_bedlevel, time_last_bedlevel = get_last_bed_levels(bedlevel_clumps, t_bedlevel_nonans)
    date_last_bedlevel = [datetime.fromtimestamp(x) for x in time_last_bedlevel]

    dbedlevel = np.diff(last_bedlevel)
    time_dbedlevel = time_last_bedlevel[1:]
    date_dbedlevel = date_last_bedlevel[1:]

    # one more refinement step to omit high freq bed level changes
    Idbed_good0 = np.where(np.diff(time_dbedlevel) > 5/6)
    # Idbed_good = (Idbed_good + np.ones(np.shape(Idbed_good))).astype(int)
    dbedlevel2 = dbedlevel[Idbed_good0]
    time_dbedlevel2 = time_dbedlevel[Idbed_good0]
    # date_dbedlevel2 = date_dbedlevel[Idbed_good0]
    date_dbedlevel2 = [datetime.fromtimestamp(x) for x in time_dbedlevel2]

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


    # find peaks within retrospective vicinity of bed level change:
    pk_times = swash_struct['data']['time_swash_peaks'][pinum] - 3*60*60
    bed_count = -1
    num_peaks = []
    d_bedlvl = []
    for bedlvl in z_bed:
        bed_count += 1

        if bed_count > 0:

            d_bedlvl.append(bedlvl - z_bed[bed_count - 1])
            t_bedlvl = t_z_bed[bed_count]
            t_last_bedlvl = t_z_bed[bed_count - 1]

            # find peaks between t and t-1:
            found_pks = np.where((pk_times < t_bedlvl) & (pk_times > t_last_bedlvl))
            # found_pks = np.where((pk_times < t_bedlvl) & (pk_times > (t_bedlvl - 20)))
            num_peaks.append(len(found_pks[0]))



    # # RMS bed level change addtion:
    # # compute in ~5 min increments with overlap?
    # dz_bed = np.diff(z_bed)
    # binlen = 60*5 # 5 mins
    # ovrlap = 0.5
    # fs = 4
    # nbins = int(len(date_z_bed[1:])/binlen/ovrlap)
    # for kk in range(nbins):
    #     RMSblc = np.sqrt(np.mean(dz_bed[nbins])**2)

    # (date_z_bed[1:], np.diff(z_bed)

    binlen = 60*4 # 5 mins
    ovrlap = 0.66
    # tmin_dz = np.min(date_dz_bed)
    # tmax_dz = np.max(date_dz_bed)
    tmin_dz = np.min(date_z_bed[1:])
    tmax_dz = np.max(date_z_bed[1:])
    tlen_dz = tmax_dz - tmin_dz
    tlen_dz_seconds = tlen_dz.total_seconds()
    nbins = int(tlen_dz_seconds/binlen/(1-ovrlap))

    RMSblc = []
    t_RMS = []

    for kk in range(nbins):
        t0 = tmin_dz + timedelta(seconds=kk*binlen*(1-ovrlap))
        t1 = t0 + timedelta(seconds=binlen)

        # compute RMS of dz within time range
        Igood = np.where(np.logical_and(np.array(date_dz_bed)>t0, np.array(date_dz_bed)<t1))
        # Igood = np.where(np.logical_and(np.array(date_z_bed[1:])>t0, np.array(date_z_bed[1:])<t1))
        if len(Igood[0])<1:
            RMSblc.append(0.)
        else:
            RMSblc.append(np.sqrt(np.mean(delta_bed[Igood]/1000)**2))
            # RMSblc.append(np.sqrt(np.mean(np.diff(z_bed)[Igood]/1000)**2))
        t_RMS.append(t1 - timedelta(seconds=binlen*0.5))



    plt.figure(num=pinum + ' dbed vs num peaks in last 20 s')
    plt.plot(num_peaks,d_bedlvl, '.')
    plt.ylabel('dz')
    plt.xlabel('num. peaks in last 20 s')



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
        ax001[0].plot(date_swash, swash, '-', color='k', Linewidth=0.25)
        # ax01[0].plot(date_swash, swash, '-', color=clr, Linewidth=0.25)

        ax001[1].plot(date_z_bed, z_bed, '.', color='k', Markersize=5)

        ax001[2].plot(date_gs, mgs, '.', color='k', Markersize=5)
        tmp_hndl = ax001[2].plot(date_gs, mgs_fit, '-', color='k', Linewidth=3)
        # handles001.extend(tmp_hndl)

    # plot MSD time series
    ax01[0].plot(date_swash, swash, '-', color=clr, Linewidth=0.25, zorder=zorder)
    # ax01[0].plot(date_swash, swash, '-', color=clr, Linewidth=0.25)

    ax01[1].plot(date_z_bed, z_bed, '.', color=clr, Markersize=3)

    ax01[2].plot(date_gs, mgs, '.', color=clr, Markersize=3)
    tmp_hndl = ax01[2].plot(date_gs, mgs_fit, '-', color=clr, Linewidth=2)
    handles01.extend(tmp_hndl)

    # ax01[3].plot(date_gs, sort, '.', color=clr, Markersize=3)
    # ax01[3].plot(date_gs, sort_fit, '-', color=clr, Linewidth=2)








    # scatter plot
    ax02[0].plot(swash_depth*1000, delta_bed,'k.')




    # FIGURES TO BE SAVED IN LOOP:


    # Methods figure: bed level change, swash peaks, with zoom in

    fig101 = plt.figure(num=pinum + ' swash with marked peaks and bed level', figsize=(6, 5))
    ax101 = fig101.add_subplot(311)
    # ax21.plot(newtt, cut_noise)
    # ax21.plot(date_good,bed_good, '.')
    # ax21.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
    ax101.plot(date_swash, swash_plus_bed, 'k', Linewidth=0.5)
    ax101.plot(date_z_bed, z_bed, 'C1.', Markersize=5)
    ax101.plot(date_swash_peaks, swash_peaks_plus_bed,'C0.', Markersize=5)
    # ax101.set_xlabel('time [UTC]')
    ax101.set_ylabel('z [m]')
    ax101.autoscale(enable=True, axis='x', tight=True)
    ax101.xaxis.set_major_formatter(xfmt)
    ax101.tick_params(direction='in', top=0, right=0)
    # ax21.invert_yaxis()
    # xfmt = md.DateFormatter('%H:%M')
    # ax21.xaxis.set_major_formatter(xfmt)
    ax101_xlims = ax101.get_xlim()

    minsoffset = 9

    if tide == '19' and pinum == 'pi71':
        ylims = [4.53, 4.68]
    elif tide == '19' and pinum == 'pi72':
        ylims = [4.53, 4.68]
    elif tide == '19' and pinum == 'pi73':
        ylims = [4.59, 4.7]
    elif tide == '19' and pinum == 'pi74':
        ylims = [4.53, 4.68]
        minsoffset = 9
    else:
        ylims = [5.02, 5.18]
        minsoffset = 15


    ax103 = fig101.add_subplot(312)
    # ax103.plot(date_dz_bed, delta_bed/1000, 'C0.')
    ax103.plot(date_z_bed[1:], np.diff(z_bed), 'k.', Markersize=5)
    ax103.plot(t_RMS, RMSblc, 'r.')
    # date_z_bed, z_bed
    # ax103.set_xlabel('time, tide ' + tide + ' [UTC]')
    ax103.set_ylabel('dz [m]')
    ax103.set_xlim(ax101_xlims)
    ax103.xaxis.set_major_formatter(xfmt)

    # date_adj = timedelta(minutes=minsoffset)
    # ax103.set_xlim([date_adj+np.min(date_swash), date_adj+np.min(date_swash) + 1/15*(np.max(date_swash) - np.min(date_swash))])
    # ax103.autoscale(enable=True, axis='y', tight=True)
    # # ax103.xaxis.set_major_formatter(xfmt)
    # # ax103.set_ylim(ylims)
    ax103.tick_params(direction='in', top=0, right=0)
    # ax22.invert_yaxis()
    # ax22.xaxis.set_major_formatter(xfmt)
    # fig101.tight_layout()


    ax102 = fig101.add_subplot(313)
    # ax22.plot(newtt, cut_noise)
    # ax22.plot(date_good,bed_good, '.')
    # ax22.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
    ax102.plot(date_swash, swash_plus_bed, 'k', Linewidth=0.5)
    ax102.plot(date_z_bed, z_bed, 'C1.')
    ax102.plot(date_swash_peaks, swash_peaks_plus_bed,'C0.')
    ax102.set_xlabel('time, tide ' + tide + ' [UTC]')
    ax102.set_ylabel('z [m]')
    # ax22.set_xlim([datetime.fromtimestamp(np.min(t_swash)), \
    #     datetime.fromtimestamp(np.min(t_swash) + 1/10*(np.max(t_swash) - np.min(t_swash)))])
    date_adj = timedelta(minutes=minsoffset)
    ax102.set_xlim([date_adj+np.min(date_swash), date_adj+np.min(date_swash) + 1/15*(np.max(date_swash) - np.min(date_swash))])
    # ax22.autoscale(enable=True, axis='x', tight=True)
    ax102.autoscale(enable=True, axis='y', tight=True)
    ax102.xaxis.set_major_formatter(xfmt)
    ax102.set_ylim(ylims)
    ax102.tick_params(direction='in', top=0, right=0)
    # ax22.invert_yaxis()
    # ax22.xaxis.set_major_formatter(xfmt)
    fig101.tight_layout()



    # Results figure: bed level change, swash peaks, dz

    fig1011 = plt.figure(num=pinum + ' swash with marked peaks and bed level, for results', figsize=(7, 4))
    ax1011 = fig1011.add_subplot(211)
    # ax21.plot(newtt, cut_noise)
    # ax21.plot(date_good,bed_good, '.')
    # ax21.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
    ax1011.plot(date_swash, swash_plus_bed, 'k', Linewidth=0.5)
    ax1011.plot(date_z_bed, z_bed, 'C1.', Markersize=5)
    ax1011.plot(date_swash_peaks, swash_peaks_plus_bed,'C0.', Markersize=5)
    # ax101.set_xlabel('time [UTC]')
    ax1011.set_ylabel('z [m]')
    ax1011.autoscale(enable=True, axis='x', tight=True)
    ax1011.xaxis.set_major_formatter(xfmt)
    ax1011.tick_params(direction='in', top=0, right=0)
    # ax21.invert_yaxis()
    # xfmt = md.DateFormatter('%H:%M')
    # ax21.xaxis.set_major_formatter(xfmt)
    ax1011_xlims = ax1011.get_xlim()

    minsoffset = 9

    if tide == '19' and pinum == 'pi71':
        ylims = [4.53, 4.68]
    elif tide == '19' and pinum == 'pi72':
        ylims = [4.53, 4.68]
    elif tide == '19' and pinum == 'pi73':
        ylims = [4.59, 4.7]
    elif tide == '19' and pinum == 'pi74':
        ylims = [4.53, 4.68]
        minsoffset = 9
    else:
        ylims = [5.02, 5.18]
        minsoffset = 15


    ax1013 = fig1011.add_subplot(212)
    # ax103.plot(date_dz_bed, delta_bed/1000, 'C0.')
    # ax1013.plot(date_z_bed[1:], np.diff(z_bed), 'k.', Markersize=5)
    ax1013.plot(date_dbedlevel2, dbedlevel2, 'k.', Markersize=5)
    ax1013.set_xlabel('time, tide ' + tide + ' [UTC]')
    ax1013.set_ylabel('dz [m]')
    ax1013.set_xlim(ax101_xlims)
    ax1013.xaxis.set_major_formatter(xfmt)

    # date_adj = timedelta(minutes=minsoffset)
    # ax103.set_xlim([date_adj+np.min(date_swash), date_adj+np.min(date_swash) + 1/15*(np.max(date_swash) - np.min(date_swash))])
    # ax103.autoscale(enable=True, axis='y', tight=True)
    # # ax103.xaxis.set_major_formatter(xfmt)
    # # ax103.set_ylim(ylims)
    ax1013.tick_params(direction='in', top=0, right=0)
    # ax22.invert_yaxis()
    # ax22.xaxis.set_major_formatter(xfmt)
    # fig101.tight_layout()

    fig1011.tight_layout()





    fig103 = plt.figure(num=pinum + '$\Delta z$ histogram')
    plt.hist(delta_bed*1000, bins=20)
    plt.xlabel('bed level change [mm]')
    plt.ylabel('frequency')


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


    fig104 = plt.figure(num=pinum + '; blc')
    ax104 = fig104.add_subplot(111)
    ax104.plot(np.diff(t_z_bed), np.diff(z_bed), '.', Markersize=5)
    ax104.set_ylabel('dz [m]')
    ax104.set_xlabel(r'dt(z) [s]')

# len(z_bed)





    # export figs
    if saveFlag == 1:
        loopdn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD','reprocessed_x15','tide'+tide, 'chunk'+chunk)

        save_figures(loopdn, pinum + '_swash_peaks_bed_level_with_dz', fig1011)
        # save_figures(loopdn, pinum + '_swash_peaks_no_bed_level', fig301)








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
# ax01[2].xaxis.set_major_formatter(plt.NullFormatter())
ax01[2].grid(alpha=gridalpha)
ax01[2].tick_params(direction='in')
ax01[2].set_ylabel('MGS [mm]')
# ax01[3].set_xlim([datemin, datemax])
# ax01[3].xaxis.set_major_formatter(xfmt)
# ax01[3].grid(alpha=gridalpha)
# ax01[3].tick_params(direction='in')
# ax01[3].set_ylabel('$M_1$ [mm]')
ax01[2].set_xlabel('time, tide ' + tide + '  [UTC]')
fig01.tight_layout()


# FIGURE 001

ax001[0].set_xlim([datemin, datemax])
ax001[0].xaxis.set_major_formatter(xfmt)
ax001[0].xaxis.set_major_formatter(plt.NullFormatter())
ax001[0].grid(alpha=gridalpha)
ax001[0].tick_params(direction='in')
# ax001[0].legend(handles01, legend_xcoords, loc="upper right")
# ax01[0].autoscale(enable=True, axis='y', tight=True)
# ax01[0].set_ylim([0, __])
ax001[0].set_ylabel('$h_{s}$ [m]')
ax001[1].set_xlim([datemin, datemax])
ax001[1].xaxis.set_major_formatter(xfmt)
ax001[1].xaxis.set_major_formatter(plt.NullFormatter())
ax001[1].grid(alpha=gridalpha)
ax001[1].tick_params(direction='in')
ax001[1].set_ylabel('$z$ [m]')
ax001[2].set_xlim([datemin, datemax])
ax001[2].xaxis.set_major_formatter(xfmt)
# ax01[2].xaxis.set_major_formatter(plt.NullFormatter())
ax001[2].grid(alpha=gridalpha)
ax001[2].tick_params(direction='in')
ax001[2].set_ylabel('MGS [mm]')
# ax01[3].set_xlim([datemin, datemax])
# ax01[3].xaxis.set_major_formatter(xfmt)
# ax01[3].grid(alpha=gridalpha)
# ax01[3].tick_params(direction='in')
# ax01[3].set_ylabel('$M_1$ [mm]')
ax001[2].set_xlabel('time, tide ' + tide + '  [UTC]')
fig001.tight_layout()



# FIGURE 02

ax02[0].set_xlabel('swash depth [mm]')
ax02[0].set_ylabel('absolute bed level change [mm]')
# fig02.tight_layout()

# FIGURE 03

delta_bed_histogram = np.squeeze(delta_bed_histogram)
delta_bed_histogram = delta_bed_histogram[np.abs(delta_bed_histogram) < 30]

# best fit of data
(mu, sigma) = norm.fit(delta_bed_histogram)

# the histogram of the data
fig03 = plt.figure(num='$\Delta z$ histogram', figsize=(5, 4))
n, bins, patches = plt.hist(delta_bed_histogram, 30, density=1, alpha=0.75)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'k--', linewidth=2)

# fig03 = plt.figure(num='$\Delta z$ histogram')
# plt.hist(delta_bed_histogram*1000, bins=30)
plt.xlabel('bed level change [mm]')
plt.ylabel('frequency [Hz]')
plt.ylim([0, 0.095])
fig03.tight_layout()


# the histogram of the data
# fig03 = plt.figure(num='$\Delta z$ histogram', figsize=(5, 4))
n, bins, patches = ax02[1].hist(delta_bed_histogram, 30, density=1, alpha=0.75)
y = mlab.normpdf(bins, mu, sigma)
l = ax02[1].plot(bins, y, 'k--', linewidth=2)

# fig03 = plt.figure(num='$\Delta z$ histogram')
# plt.hist(delta_bed_histogram*1000, bins=30)
ax02[1].set_xlabel('bed level change [mm]')
ax02[1].set_ylabel('frequency [Hz]')
fig02.tight_layout()



# export figs
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD','reprocessed_x10','tide'+tide, 'chunk'+chunk)

    save_figures(savedn, 'MSD_timeseries', fig01)
    save_figures(savedn, 'MSD_timeseries_pi4', fig001)
    # # save_figures(savedn, 'delta_bed_level_swash_depth_scatter', fig02)
    save_figures(savedn, 'bed_level_swash_depth_scatter_and_histogram_'+'tide'+tide+'_chunk'+chunk, fig02)
    save_figures(savedn, 'delta_bed_level_histogram', fig03)
