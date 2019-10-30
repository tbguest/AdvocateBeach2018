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
import matplotlib.dates as md
import sys
# from .data.regresstools import lowess
from src.data.regresstools import lowess
from src.data.regresstools import loess_1d
from datetime import datetime
import time
from scipy.ndimage import label

from pathlib import Path


%matplotlib qt5

# change default font size
plt.rcParams.update({'font.size': 12})


def utime2yearday(unixtime):

    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday


def findpeaks(bed_interp, cut_noise, newtt):
    '''
    Takes the isolated swash and bed level data
    and finds peaks in swash depth based on selection criteria.
    '''

    t_cutoff = 5. # no. of samples below which swash chunks are disregarded
    h_thresh = 15. # mm

    nan2zero = np.copy(cut_noise)
    nan2zero[np.isnan(nan2zero)] = 0.

    # cycle through all swash chunks separated by nan values
    labeled_array, num_features = label(nan2zero)

    Iall_max = []

    for seg in range(1, num_features):
        Iseg = np.argwhere(labeled_array==seg)

        # local maximum
        Ilocmax = np.argmin(nan2zero[Iseg])

        # length cutoff
        if len(Iseg) > t_cutoff:
            # no endpoints
            if Ilocmax != 0 and Ilocmax != len(Iseg)-1:
                # |max - bed| > some value
                if np.abs(bed_interp[Ilocmax + Iseg[0]] - cut_noise[Ilocmax + Iseg[0]]) > h_thresh:
                    if cut_noise[Ilocmax + Iseg[0]] > 500.:
                        Iall_max.append(Ilocmax + Iseg[0])

    # reduce to 1d
    Iall_max = np.squeeze(Iall_max)

    return Iall_max


def bed_change(newtt, cut_noise, Iall_max, date_good, bed_good):

    prox = 5

    dbed = []
    swash_dep = []
    t_swash_dep = []

# peak = 266

    for peak in Iall_max:
        # print(peak)
        # keep if there exists a bed signal within x samples prior to and following
        # peak = Iall_max[0]
        tdiff = newtt[peak] - date_good

        pre = np.copy(tdiff)
        pre[pre < 0] = np.float(np.nan)
        mn_pre = np.nanargmin(np.abs(pre))

        post = np.copy(tdiff)
        post[post > 0] = np.float(np.nan)
        mn_post = np.nanargmin(np.abs(post))

        if np.abs(tdiff[mn_pre]) < prox and np.abs(tdiff[mn_post]) < prox:
            dbed.append(bed_good[mn_post] - bed_good[mn_pre])
            swash_dep.append(bed_good[mn_pre] - cut_noise[peak])
            t_swash_dep.append(date_good[mn_pre])

    return np.squeeze(dbed), np.squeeze(swash_dep), np.squeeze(t_swash_dep)


def extract_bed(tt, raw):
    '''Accepts raw range data and time stamps,

    '''

    # time and range thresholds for bed extraction
    t_thresh = 9 # samples
    d_thresh = 5 # mm; roughly commensurate with noise floor

    # initialize
    bed_vec = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

    # populate bed vector
    # Checks the 9 previous points to see if they meet threshold criterion
    for ii in range(t_thresh, len(raw)):
        for jj in range(1, t_thresh):
            if abs(raw[ii] - raw[ii-jj]) > d_thresh:
                bval = float('nan')
                break
            bval = np.mean(raw[ii-t_thresh+1:ii])
        bed_vec.append(bval)

    # clean nans
    bed_vec = np.array(bed_vec)
    date1 = np.array(tt)
    Ibed = np.argwhere(~np.isnan(bed_vec))#.item()
    bed_good = bed_vec[Ibed]
    date_good = tt[Ibed]

    bed_with_nans = bed_vec
    date_with_nans = date1

    return np.squeeze(bed_good), np.squeeze(date_good), np.squeeze(bed_with_nans), np.squeeze(date_with_nans)


def get_swash(bed_good, date_good, raw, tt):
    '''separates swash data from bed signal
    '''

    # gap filling:
    newtt = np.linspace(np.min(date_good), np.max(date_good), len(raw))
    bed_interp = np.interp(newtt, np.squeeze(date_good), np.squeeze(bed_good))
    raw_interp = np.interp(newtt, np.squeeze(tt), np.squeeze(raw))

    # separate swash from bed signal (with buffer region)
    cut_noise = np.copy(raw_interp)
    cut_noise[cut_noise > bed_interp-2] = np.float(np.nan)

    # find all maxima in new, separated swash signal
    Iall_max = findpeaks(bed_interp, cut_noise, newtt)

    return newtt, bed_interp, cut_noise, Iall_max



# homechar = "C:\\" # windows
homechar = os.path.expanduser("~") # linux
# ext drive:
homechar_ext = os.path.join('/media','tristan2','Advocate2018_backup2')

tide = '15'
tide = '19'
# tide = '27'

# chunk = '2'

post = 1.7 # survey post height

rngdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'tide' + tide)

chunks = sorted(os.listdir(rngdir))

for chunk in chunks:

    # # pi locations
    # pifile = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
    #                            'processed', 'GPS', 'array', 'tide' + tide, \
    #                            'array_position' + chunk + '.npy')

    if tide == '27':
        pinums = ['pi71', 'pi73', 'pi74']
    else:
        pinums = ['pi71', 'pi72', 'pi73', 'pi74']

    # load:
    # pilocs = np.load(pifile, allow_pickle=True).item()
    bbb = np.load(os.path.join(rngdir, chunk), allow_pickle=True).item()

    # swash data
    swash = {}
    swash_plus_bed = {}
    time_swash = {}
    date_swash = {}
    swash_peaks = {}
    swash_peaks_plus_bed = {}
    time_swash_peaks = {}
    date_swash_peaks = {}
    swash_depth = {}
    time_swash_depth = {}
    date_swash_depth = {}
    delta_bed = {}

    # bed level data
    bed = {}
    time_bed = {}
    date_bed = {}

    bed_level = {}
    date_bed_level = {}
    time_bed_level = {}

    # structures to be saved
    swash_data = {}
    bed_level_data = {}

    # metadata
    swash_data['comments'] = ['swash [m]: swash-only signal',
                            'swash_plus_bed [m]: swash-only signal, without subtracted bed',
                            'time_swash: unix timestamp for each swash point',
                            'date_swash: date object for each swash point',
                            'swash_peaks [m]: swash depth maxima',
                            'swash_peaks_plus_bed [m]: swash depth maxima, without subtracted bed',
                            'time_swash_peaks: unix timestamp for each swash maximum',
                            'date_swash_peaks: date object for each swash maximum',
                            'swash_depth [m]: swash depth maxima in proximity of bed level change',
                            'time_swash_depth: unix timestamp for each swash depth',
                            'date_swash_depth: date object for each swash depth',
                            'delta_bed [m]: bed level change corresponding to swash_depth']

    bed_level_data['comments'] = ['bed [m]: bed level signal (uncorrected for pi locations)',
                            'time_bed: unix timestamp for each bed point',
                            'date_bed: date object for each bed point',
                            'dbed: change in bed level based on last bed point in each sequence',
                            'time_dbed: time associated with dbed']

    for pino in pinums:

        # pino = 'pi74'

        # range data
        raw = bbb[pino]['raw_range']
        tt = bbb[pino]['tvec_raw']

        # extract bed
        bed_good, date_good, bed_with_nans, date_with_nans = extract_bed(tt, raw)

        # extract swash
        newtt, bed_interp, cut_noise, Iall_max = get_swash(bed_good, date_good, raw, tt)

        # all peak swash dpeths
        all_peaks = np.abs(cut_noise[Iall_max]-bed_interp[Iall_max])/1000
        all_peaks_plus_bed = np.abs(cut_noise[Iall_max])/1000

        # compute bed change and swash depth stats
        dbed, swash_dep, t_swash_dep = bed_change(newtt, cut_noise, Iall_max, date_good, bed_good)

        # convert unix times to dates for plotting
        tt_date = [datetime.fromtimestamp(x) for x in newtt]
        tt_date_good = [datetime.fromtimestamp(x) for x in date_good]
        tt_Iall_max = [datetime.fromtimestamp(x) for x in newtt[Iall_max]]
        date_swash_dep = [datetime.fromtimestamp(x) for x in t_swash_dep]
        tt_date_with_nans = [datetime.fromtimestamp(x) for x in date_with_nans]

        # consolidate vbles to save
        swash[pino] = np.abs(cut_noise-bed_interp)/1000
        swash_plus_bed[pino] = cut_noise/1000
        time_swash[pino] = newtt
        date_swash[pino] = tt_date

        swash_peaks[pino] = all_peaks
        swash_peaks_plus_bed[pino] = all_peaks_plus_bed
        time_swash_peaks[pino] = newtt[Iall_max]
        date_swash_peaks[pino] = tt_Iall_max

        swash_depth[pino] = swash_dep/1000.
        time_swash_depth[pino] = t_swash_dep
        date_swash_depth[pino] = date_swash_dep
        delta_bed[pino] = dbed/1000.

        bed[pino] = bed_good/1000.
        time_bed[pino] = date_good
        date_bed[pino] = tt_date_good

        # bed level data with nans included - leaving more flexibility for bed level change computation
        bed_level[pino] = bed_with_nans/1000.
        date_bed_level[pino] = date_with_nans
        time_bed_level[pino] = tt_date_with_nans




    swash_data['data'] = {'swash':swash, 'swash_plus_bed':swash_plus_bed, 'time_swash':time_swash, 'date_swash':date_swash, \
                        'swash_peaks':swash_peaks, 'swash_peaks_plus_bed':swash_peaks_plus_bed, 'time_swash_peaks':time_swash_peaks, \
                        'date_swash_peaks':date_swash_peaks, 'swash_depth':swash_depth, \
                        'time_swash_depth':time_swash_depth, 'date_swash_depth':date_swash_depth, \
                        'delta_bed':delta_bed}

    bed_level_data['data'] = {'bed':bed, 'time_bed':time_bed, 'date_bed':date_bed, \
                        'bed_level':bed_level, 'time_bed_level':time_bed_level, \
                        'date_bed_level':date_bed_level}


    #### EXPORT DATA DICTS ###############

    saveFlag = 1

    if saveFlag == 1:
        swashdn = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                                   'processed', 'range_data', 'swash', 'tide' + tide)

        beddn = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                                   'processed', 'range_data', 'bed_level', 'tide' + tide)

        if not os.path.exists(swashdn):
            try:
                os.makedirs(swashdn)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        if not os.path.exists(beddn):
            try:
                os.makedirs(beddn)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


        np.save(os.path.join(swashdn, chunk), swash_data)
        np.save(os.path.join(beddn, chunk), bed_level_data)

    ##################################



### FIGURES ############

fig1 = plt.figure(1)
# plt.plot(bbb['pi71']['tvec_raw'], bbb['pi71']['raw_range'])
plt.plot(date_good,bed_good, '.')
plt.plot(newtt,cut_noise)
ax = plt.gca()
ax.invert_yaxis()


fig2 = plt.figure(2)
ax21 = fig2.add_subplot(211)
# ax21.plot(newtt, cut_noise)
# ax21.plot(date_good,bed_good, '.')
# ax21.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
ax21.plot(tt_date, cut_noise)
ax21.plot(tt_date_good,bed_good, '.')
ax21.plot(tt_Iall_max, cut_noise[Iall_max],'.')
ax21.set_xlabel('time [UTC]')
ax21.set_ylabel('range [mm]')
ax21.autoscale(enable=True, axis='x', tight=True)
ax21.invert_yaxis()
xfmt = md.DateFormatter('%H:%M')
ax21.xaxis.set_major_formatter(xfmt)

ax22 = fig2.add_subplot(212)
# ax22.plot(newtt, cut_noise)
# ax22.plot(date_good,bed_good, '.')
# ax22.plot(newtt[Iall_max], cut_noise[Iall_max],'.')
ax22.plot(tt_date, cut_noise)
ax22.plot(tt_date_good,bed_good, '.')
ax22.plot(tt_Iall_max, cut_noise[Iall_max],'.')
ax22.set_xlabel('time [UTC]')
ax22.set_ylabel('range [mm]')
ax22.set_xlim([datetime.fromtimestamp(np.min(newtt)), \
    datetime.fromtimestamp(np.min(newtt) + 1/10*(np.max(newtt) - np.min(newtt)))])
# ax22.autoscale(enable=True, axis='x', tight=True)
ax22.autoscale(enable=True, axis='y', tight=True)
ax22.invert_yaxis()
ax22.xaxis.set_major_formatter(xfmt)
fig2.tight_layout()


fig3 = plt.figure(3)
plt.plot(swash_dep, np.abs(dbed),'.')
plt.xlabel('swash depth [mm]')
plt.ylabel('absolute bed level change [mm]')


fig4 = plt.figure(4)
plt.hist(np.squeeze(dbed), bins=16)
plt.xlabel('bed level change [mm]')
plt.ylabel('occurrences')


fig5 = plt.figure(5)
# plt.plot(bbb['pi71']['tvec_raw'], bbb['pi71']['raw_range'])
plt.plot(tt_date, np.abs(cut_noise-bed_interp)/1000)
# plt.plot(date_swash_dep, swash_dep, '.')
plt.plot(tt_Iall_max, np.abs(cut_noise[Iall_max]-bed_interp[Iall_max])/1000, '.')
# plt.plot(newtt,cut_noise)
# ax = plt.gca()
# ax.invert_yaxis()


savePlotFlag = 0

if savePlotFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports',\
        'figures','swash','tide' + tide, 'chunk' + chunk)
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # fig1.savefig(os.path.join(savedn, 'array_timeseries.png'), dpi=1000, transparent=True)newtt
    # fig1.savefig(os.path.join(savedn, 'array_timeseries.pdf'), dpi=None, transparent=True)

    fig2.savefig(os.path.join(savedn, pino + '_range_timeseries.png'), dpi=1000, transparent=True)
    fig2.savefig(os.path.join(savedn, pino + 'range_timeseries.pdf'), dpi=None, transparent=True)

    fig3.savefig(os.path.join(savedn, pino + 'bed_change_swash_depth.png'), dpi=1000, transparent=True)
    fig3.savefig(os.path.join(savedn, pino + 'bed_change_swash_depth.pdf'), dpi=None, transparent=True)

    fig4.savefig(os.path.join(savedn, pino + 'bed_change_histogram.png'), dpi=1000, transparent=True)
    fig4.savefig(os.path.join(savedn, pino + 'bed_change_histogram.pdf'), dpi=None, transparent=True)

    # fig2.savefig(os.path.join(savedn, 'transport_stats.png'), dpi=1000, transparent=True)
    # fig2.savefig(os.path.join(savedn, 'transport_stats.pdf'), dpi=None, transparent=True)
