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

    return dbed, swash_dep

        # keep if no other peaks fall in between this one and bed observations
#
# plt.figure()
# plt.plot(np.abs(bar))



tide = '27'
chunk = '4'
pino = 'pi74'

post = 1.7 # survey post height

homechar = "C:\\"

rngdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'bed_level', 'tide' + tide)

# pi locations
pifile = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'GPS', 'array', 'tide' + tide, \
                           'array_position' + chunk + '.npy')
pilocs = np.load(pifile).item()

bbb = np.load(os.path.join(rngdir, 'chunk' + chunk + '.npy')).item()
#
# plt.figure()
# plt.plot(bbb['pi71']['tvec_raw'], bbb['pi71']['raw_range'])
# plt.plot(bbb['pi71']['tvec'], bbb['pi71']['range'], '-')
# ax = plt.gca()
# ax.invert_yaxis()


# try new bed extraction criteria

raw = bbb[pino]['raw_range']
tt = bbb[pino]['tvec_raw']


t_thresh = 7 # samples
d_thresh = 5 # mm; roughly commensurate with noise floor


# bed_vec = np.zeros(len(raw))
#
# for ii in range(t_thresh, len(raw)):
#     for jj in range(1, t_thresh):
#         if abs(raw[ii-jj+1] - raw[ii-jj]) > d_thresh:
#             break
#


bed_vec = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

for ii in range(t_thresh, len(raw)):
    for jj in range(1, t_thresh):
        if abs(raw[ii] - raw[ii-jj]) > d_thresh:
            bval = float('nan')
            break
        bval = np.mean(raw[ii-t_thresh+1:ii])
    bed_vec.append(bval)

bed_vec = np.array(bed_vec)
date1 = np.array(tt)
Ibed = np.argwhere(~np.isnan(bed_vec))#.item()
bed_good = bed_vec[Ibed]
date_good = tt[Ibed]

raw_bed_level = np.array([date_good, bed_good])
raw_range = np.array([tt, raw])

#
# plt.figure()
# plt.plot(bbb['pi71']['tvec_raw'], bbb['pi71']['raw_range'])
# plt.plot(date_good, bed_good, 'o')
# plt.plot(bbb['pi71']['tvec'], bbb['pi71']['range'], '.')
# ax = plt.gca()
# ax.invert_yaxis()


newtt = np.linspace(np.min(date_good), np.max(date_good), len(raw))
bed_interp = np.interp(newtt, np.squeeze(date_good), np.squeeze(bed_good))
raw_interp = np.interp(newtt, np.squeeze(tt), np.squeeze(raw))

# tt_date = [datetime.fromtimestamp(x).strftime('%d/%m/%Y %H:%M:%S.%f') for x in newtt]
tt_date = [datetime.fromtimestamp(x) for x in newtt]
tt_date_good = [datetime.fromtimestamp(x) for x in date_good]

# print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
# plt.figure()
# plt.plot(bbb['pi71']['tvec_raw'], bbb['pi71']['raw_range'])
# plt.plot(newtt,bed_interp)
# plt.plot(newtt,raw_interp)
# ax = plt.gca()
# ax.invert_yaxis()


cut_noise = np.copy(raw_interp)
cut_noise[cut_noise > bed_interp-2] = np.float(np.nan)
# cut_noise[cut_noise-bed_interp > -2] = np.float(np.nan)

# cut_noise[ np.array([j > 709 if ~np.isnan(j) else False for j in cut_noise], dtype=bool) ] = 709

fig1 = plt.figure(1)
# plt.plot(bbb['pi71']['tvec_raw'], bbb['pi71']['raw_range'])
plt.plot(date_good,bed_good, '.')
plt.plot(newtt,cut_noise)
ax = plt.gca()
ax.invert_yaxis()


Iall_max = findpeaks(bed_interp, cut_noise, newtt)

tt_Iall_max = [datetime.fromtimestamp(x) for x in newtt[Iall_max]]

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


dbed, swash_dep = bed_change(newtt, cut_noise, Iall_max, date_good, bed_good)

fig3 = plt.figure(3)
plt.plot(swash_dep, np.abs(dbed),'.')
plt.xlabel('swash depth [mm]')
plt.ylabel('absolute bed level change [mm]')


fig4 = plt.figure(4)
plt.hist(np.squeeze(dbed), bins=16)
plt.xlabel('bed level change [mm]')
plt.ylabel('occurrences')




saveFlag = 0
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports',\
        'figures','swash','tide' + tide, 'chunk' + chunk)
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # fig1.savefig(os.path.join(savedn, 'array_timeseries.png'), dpi=1000, transparent=True)
    # fig1.savefig(os.path.join(savedn, 'array_timeseries.pdf'), dpi=None, transparent=True)

    fig2.savefig(os.path.join(savedn, pino + '_range_timeseries.png'), dpi=1000, transparent=True)
    fig2.savefig(os.path.join(savedn, pino + 'range_timeseries.pdf'), dpi=None, transparent=True)

    fig3.savefig(os.path.join(savedn, pino + 'bed_change_swash_depth.png'), dpi=1000, transparent=True)
    fig3.savefig(os.path.join(savedn, pino + 'bed_change_swash_depth.pdf'), dpi=None, transparent=True)

    fig4.savefig(os.path.join(savedn, pino + 'bed_change_histogram.png'), dpi=1000, transparent=True)
    fig4.savefig(os.path.join(savedn, pino + 'bed_change_histogram.pdf'), dpi=None, transparent=True)

    # fig2.savefig(os.path.join(savedn, 'transport_stats.png'), dpi=1000, transparent=True)
    # fig2.savefig(os.path.join(savedn, 'transport_stats.pdf'), dpi=None, transparent=True)


# datetime(newtt[0])
# tt_date = [datetime.fromtimestamp(x).strftime('%d/%m/%Y %H:%M:%S.%f') for x in newtt]

# plt.figure()
# plt.plot(tt_date, raw_interp)
# ax=plt.gca()
# xfmt = md.DateFormatter('%H:%M')
# ax.xaxis.set_major_formatter(xfmt)
