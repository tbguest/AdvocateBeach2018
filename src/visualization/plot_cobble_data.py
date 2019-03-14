'''
13 March 2019
Tristan Guest
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal
import time
from datetime import datetime
# %matplotlib qt5

# change default font size
plt.rcParams.update({'font.size': 12})


plt.close("all")
saveFlag = 1

def smooth_pressure(dep):
    ''' accepts a depth signal, p, for a given tide, and returns a smoothed
    curve, rescaled by the beach slope to give mean cross-shore shoreline
    position [m].
    '''

    b, a = signal.butter(2, 0.0005)
    mean_shoreline = signal.filtfilt(b, a, dep)/0.12 # assumed beach slope of 0.12

    return mean_shoreline


def utime2yearday(unixtime):

    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday


def binTrajectoryData(stones):
    '''Accepts a list of integer stone indices. Stone trajectory data are
    organized by bins, where each bin is some fraction of the original image
    width. Trajectories > some threshold are added to a bin determined by their
    starting position.
    '''

    bins = {}
    net_dx = []
    net_dy = []
    abs_dx_sum = []
    abs_dy_sum = []
    cumul_trans = []

    for stone in stones:

        dx = np.diff(stones[int(stone)]['x'])
        dy = np.diff(stones[int(stone)]['y'])

        net_dx.append(np.sum(dx))
        net_dy.append(np.sum(dy))

        cumul_trans.append(np.sum(np.sqrt(dx**2 + dy**2)))

        abs_dx_sum.append(np.sum(np.abs(dx)))
        abs_dy_sum.append(np.sum(np.abs(dy)))

        Itransport = np.where(np.abs(dx) > transport_thresh)

        for k in range(len(Itransport[0])):

            # binning in image-centric reference frame
            xpix = stones[int(stone)]['x_pixel']
            binx = np.int(xpix[Itransport[0][k]]/1000*nbins) # i think this gives the right starting point...
            # append to dict bin
            if binx not in bins:
                bins[binx] = [dx[Itransport[0][k]]]
            else:
                bins.setdefault(binx, []).append(dx[Itransport[0][k]]) # worked at this one for a while...

    return bins, net_dx, net_dy, abs_dx_sum, abs_dy_sum, cumul_trans


# VARIABLE DEFINITIONS

homechar = "C:\\"
stone_class = 'yellows'
# for tide 19, position 1: scaling = 421.0550841144531 pix/m
tide = "tide19"
navg = 10 # 10 frames were averaged
nbins = 8
transport_thresh = 0.01 # 1cm -- for omitting small changes in cum sum of transport
positions = ["position1", "position2", "position3", "position4"]

# LOAD DATA

# pressure data for determining mean shoreline
pressurefn = os.path.join(homechar,'Projects','AdvocateBeach2018','data','interim','pressure',tide + '.npy')
p = np.load(pressurefn).item()
tt = p['t']
dep = p['d']
mean_shoreline = smooth_pressure(dep) # not yet aligned with swash

# INITIALIZE

# initialize plot before loop
fig1, ax1 = plt.subplots(nrows=1, ncols=1,num='camera positions with tide', figsize=(5,3.8))
ax1.plot(tt, mean_shoreline, 'k', Linewidth=2)
ax1.plot(tt, mean_shoreline + 0.68*2, 'r', Linewidth=2)
ax1.plot(tt, mean_shoreline - 0.68*2, 'r', Linewidth=2)
ax1.set_ylabel('cross shore distance [m]')
ax1.set_xlabel('yearday')
ax1.set_xlim([295.553, 295.655])
ax1.set_ylim([37, 54])
fig1.tight_layout()

# initialize transport stat vectors
mean_xtransport = np.zeros(len(positions))
mean_ytransport = np.zeros(len(positions))
std_xtransport = np.zeros(len(positions))
std_ytransport = np.zeros(len(positions))
abs_xtransport = np.zeros(len(positions))
abs_ytransport = np.zeros(len(positions))
std_abs_xtransport = np.zeros(len(positions))
std_abs_ytransport = np.zeros(len(positions))
mean_cumul_trans = np.zeros(len(positions))
std_cumul_trans = np.zeros(len(positions))

# fig3, [ax31,ax32] = plt.subplots(nrows=2, ncols=1, num='station 1: binned transport stats', gridspec_kw = {'height_ratios':[1, 3]})
# fig4, [ax41,ax42] = plt.subplots(nrows=2, ncols=1, num='station 2: binned transport stats', gridspec_kw = {'height_ratios':[1, 3]})
# fig5, [ax51,ax52] = plt.subplots(nrows=2, ncols=1, num='station 3: binned transport stats', gridspec_kw = {'height_ratios':[1, 3]})
# fig6, [ax61,ax62] = plt.subplots(nrows=2, ncols=1, num='station 4: binned transport stats', gridspec_kw = {'height_ratios':[1, 3]})

for position in positions:

    # for setting axis limits on swash + trajectory figures
    if position == 'position1':
        offset = 49.96877597308448
        scaling = 421.0550841144531
    elif position == 'position2':
        offset = 45.17029088878186
        scaling = 458.05166513842414
    elif position == 'position3':
        offset = 41.03177921964147
        scaling = 472.43762017669604
    elif position == 'position4':
        offset = 37.85738057659421
        scaling = 436.65137206616646

    # LOAD DATA

    # load swash timeseries data
    dn_swash = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', \
                'data', 'processed', 'swash', tide, 'camera_' + position)
    swash_ts = np.load(os.path.join(dn_swash, 'swash_timeseries.npy')).item()
    swash_std = swash_ts['stdev']

    # load cobble transport data
    dn_cobble = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', \
                'data', 'processed', 'cobble_tracking', tide, 'camera_' + position)
    stones = np.load(os.path.join(dn_cobble, stone_class + '.npy')).item()


    ## bin tansport data
    bins, net_dx, net_dy, abs_dx_sum, abs_dy_sum, cumul_trans  = binTrajectoryData(stones)

    # this funky indexing is just 0-3, without having to add a counter
    mean_xtransport[np.int(position[-1]) - 1] = np.mean(np.array(net_dx))
    mean_ytransport[np.int(position[-1]) - 1] = np.mean(np.array(net_dy))
    std_xtransport[np.int(position[-1]) - 1] = np.std(np.array(net_dx))
    std_ytransport[np.int(position[-1]) - 1] = np.std(np.array(net_dy))

    abs_xtransport[np.int(position[-1]) - 1] = np.mean(abs_dx_sum)
    abs_ytransport[np.int(position[-1]) - 1] = np.mean(abs_dy_sum)
    std_abs_xtransport[np.int(position[-1]) - 1] = np.std(abs_dx_sum)
    std_abs_ytransport[np.int(position[-1]) - 1] = np.std(abs_dy_sum)

    mean_cumul_trans[np.int(position[-1]) - 1] = np.mean(np.array(cumul_trans))
    std_cumul_trans[np.int(position[-1]) - 1] = np.std(np.array(cumul_trans))

    # operations on binned data
    binmean = np.zeros(nbins)
    binstd = np.zeros(nbins)
    bincount = np.zeros(nbins)
    imgbins = np.arange(nbins)
    for n in range(nbins):
        if n in bins:
            binmean[n] = np.mean(bins[n])
            binstd[n] = np.std(bins[n])
            bincount[n] = len(bins[n])


    ## IN-LOOP PLOTS

    # plot swash time series for each position on mean shoreline plot
    if position == 'position2':
        ax1.plot(swash_ts['tvec'][:-450], swash_ts['0600'][:-450], c='C0', Linewidth=0.5)
    else:
        ax1.plot(swash_ts['tvec'], swash_ts['0600'], c='C0', Linewidth=0.5)


    # plot binned data
    fi, axs = plt.subplots(nrows=1, ncols=2, num='station ' + position[-1] + ': binned transport stats', \
                gridspec_kw = {'width_ratios':[3, 1]}, figsize=(5.5,4))
    axs[0].plot(binmean, imgbins, 'ko')
    axs[0].plot([binmean-binstd, binmean+binstd], [imgbins, imgbins], 'k-')
    axs[0].plot(np.zeros(50), np.linspace(-0.25, nbins-1 + 0.25, 50), 'k--')
    axs[0].set_xlabel('cross-shore transport distance [m]')
    axs[0].set_ylabel('cross-shore bin')
    axs[0].set_ylim(-0.25, nbins-1 + 0.25)
    # axs[0].invert_yaxis()

    axs[1].plot(bincount, imgbins,'ko')
    axs[1].set_xlabel('count')
    axs[1].set_xlim(-1, np.max(bincount)+0.1*(np.max(bincount)))
    axs[1].set_ylim(-0.25, nbins-1 + 0.25)
    axs[1].yaxis.set_major_formatter(plt.NullFormatter())
    # axs[1].invert_yaxis()
    fi.tight_layout()


    # plot Trajectories
    fi2, axs2 = plt.subplots(nrows=1, ncols=1, num='station ' + position[-1] + ': cross-shore trajectories', figsize=(5.5,4))
    if position == 'position2':
        axs2.plot(swash_ts['tvec'][:-450], swash_ts['0600'][:-450], c='grey', Linewidth=0.5, alpha=.5)
    else:
        axs2.plot(swash_ts['tvec'], swash_ts['0600'], c='grey', Linewidth=0.5, alpha=.5)
    # for setting ax lims:
    mint = 1000
    maxt = 0
    minx = 1000
    maxx = 0
    for k in stones:
        axs2.plot(stones[k]['t'], stones[k]['x'])
        if np.min(stones[k]['t']) < mint:
            mint = np.min(stones[k]['t'])
        if np.min(stones[k]['t']) > maxt:
            maxt = np.max(stones[k]['t'])
        # if np.min(stones[k]['x']) < minx:
        #     minx = np.min(stones[k]['x'])
        # if np.min(stones[k]['x']) > maxx:
        #     maxx = np.max(stones[k]['x'])
    axs2.set_xlabel('yearday')
    axs2.set_ylabel('cross-shore coordinate [m]')
    # axs2.invert_yaxis()
    minx = offset
    maxx = 1000*1640/1000/scaling + offset
    axs2.set_ylim([minx, maxx])
    axs2.get_xaxis().get_major_formatter().set_useOffset(False)
    axs2.xaxis.set_major_locator(plt.MaxNLocator(4))
    axs2.set_xlim([mint, maxt])
    fi2.tight_layout()


    # EXPORT PLOTS

    if saveFlag == 1:

        savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','cobble_transport',tide)
        if not os.path.exists(savedn):
            try:
                os.makedirs(savedn)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        fi.savefig(os.path.join(savedn, 'binned_transport_stats_' + position + '.png'), dpi=1000, transparent=True)
        fi.savefig(os.path.join(savedn, 'binned_transport_stats_' + position + '.pdf'), dpi=None, transparent=True)

        fi2.savefig(os.path.join(savedn, 'cross_shore_trajectories_' + position + '.png'), dpi=1000, transparent=True)
        fi2.savefig(os.path.join(savedn, 'cross_shore_trajectories_' + position + '.pdf'), dpi=None, transparent=True)


# OUT-OF-LOOP PLOTS

# stats for all camera stations
fig2, (ax21,ax22) = plt.subplots(nrows=1, ncols=2,num='bulk transport stats', figsize=(6.5,4.0))

yoffset = 0.1
ax21.plot(mean_xtransport, np.linspace(1,4,4), 'ko')
ax21.plot(mean_ytransport, np.linspace(1,4,4) + yoffset, 'ro')
ax21.plot([mean_xtransport+std_xtransport, mean_xtransport-std_xtransport], \
            [np.linspace(1,4,4), np.linspace(1,4,4)], 'k')
ax21.plot([mean_ytransport+std_ytransport, mean_ytransport-std_ytransport], \
            [np.linspace(1,4,4) + yoffset, np.linspace(1,4,4) + yoffset], 'r')
ax21.plot(np.zeros(100), np.linspace(0.8, 4.2 + yoffset, 100), 'k--')
ax21.invert_yaxis()
ax21.set_ylim(4.2, 0.8)
ax21.set_ylabel('station')
ax21.set_xlabel('net transport/cobble [m]')
ax21.yaxis.set_major_locator(plt.MultipleLocator(1))

ax22.plot(abs_xtransport, np.linspace(1,4,4), 'ko')
ax22.plot(abs_ytransport, np.linspace(1,4,4) + yoffset, 'ro')
ax22.plot([abs_xtransport+std_abs_xtransport, abs_xtransport-std_abs_xtransport], \
            [np.linspace(1,4,4), np.linspace(1,4,4)], 'k')
ax22.plot([abs_ytransport+std_abs_ytransport, abs_ytransport-std_abs_ytransport], \
            [np.linspace(1,4,4) + yoffset, np.linspace(1,4,4) + yoffset], 'r')
ax22.invert_yaxis()
ax22.set_ylim(4.2, 0.8)
ax22.set_xlabel('cumulative transport/cobble [m]')
ax22.legend(['cross-shore', 'longshore'])
# ax22.yaxis.set_major_locator(plt.NullLocator())
ax22.yaxis.set_major_formatter(plt.NullFormatter())
ax22.yaxis.set_major_locator(plt.MultipleLocator(1))

fig2.tight_layout()


# EXPORT PLOTS

if saveFlag == 1:

    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','cobble_transport',tide)
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig1.savefig(os.path.join(savedn, 'swash_stations_tide.png'), dpi=1000, transparent=True)
    fig1.savefig(os.path.join(savedn, 'swash_stations_tide.pdf'), dpi=None, transparent=True)

    fig2.savefig(os.path.join(savedn, 'transport_stats.png'), dpi=1000, transparent=True)
    fig2.savefig(os.path.join(savedn, 'transport_stats.pdf'), dpi=None, transparent=True)
