
# coding: utf-8

"""
Created on Fri Jan  4 10:34:20 2019

@author: Tristan Guest

Extracts bed level from the altimeter time series using a simple criterion.
Saves the full raw range series, the full extracted bed series, bed series
divided in to chunks, and a smoothed bed series for each chunk. The smooth
time series chunks are used to reduce noise when computing mm/pixel in dgs
calculation. One more pass over these data is done by ____ to save the chunks
in more usable form.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
from scipy import interpolate


#### DEFUNCT ################

def build_unixtime(year, month, day, hour, minute, second):

    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), 0)
    utime = time.mktime(dt.timetuple()) + second

    return utime


find_points_of_interest = False

## tide ?
## 15 [no timestamps!]
#fn1 = "rng10152018_pi71.001"
#fn2 = "rng10152018_pi72.001"
#fn3 = "rng10152018_pi73.001"
#fn4 = "rng10152018_pi74.001"
#dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\15_10_2018\\PM\\"


# tide 13
# 20
#fn1 = "sonar71_2018-10-20-13_08.dat"
#fn1 = "sonar72_2018-10-20-13_08.dat"
#fn1 = "sonar73_2018-10-20-13_08.dat"
#fn1 = "sonar74_2018-10-20-13_08.dat"
#dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\20_10_2018\\AM\\"
#tide = '13'


# tide 15
# 21 - cusp day
#fn1 = "sonar71_2018-10-21-13_01.dat"
#fn1 = "sonar72_2018-10-21-13_01.dat"
#fn1 = "sonar73_2018-10-21-13_01.dat"
#fn1 = "sonar74_2018-10-21-13_01.dat"
#dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\21_10_2018\\AM\\"
#tide = '15'


# tide 19
# 23 - cusp day
#fn1 = "sonar71_2018-10-23-13_37.dat"
#fn1 = "sonar72_2018-10-23-13_37.dat"
#fn1 = "sonar73_2018-10-23-13_37.dat"
#fn1 = "sonar74_2018-10-23-13_37.dat"
#dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\23_10_2018\\AM\\"
#tide = '19'

## tide 21
## 24
#fn1 = "sonar71_2018-10-24-14_14.dat"
#fn1 = "sonar72_2018-10-24-14_14.dat"
#fn1 = "sonar73_2018-10-24-14_14.dat"
#fn1 = "sonar74_2018-10-24-14_15.dat"
#dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\24_10_2018\\PM\\"
#tide = '21'

## tide 27
## 27
fn1 = "sonar71_2018-10-27-16_16.dat"
#fn1 = "sonar73_2018-10-27-16_16.dat"
#fn1 = "sonar74_2018-10-27-16_16.dat"
dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\27_10_2018\\"
tide = '27'

# unix tstapmp of main run start: 1540303640
# end: 1540308900


timesdir = os.path.join("C:\\", "Projects", "AdvocateBeach2018", "data", \
                        "interim", "range_data", "start_and_end_times", \
                        "tide" + tide + ".npy")

# defines start and end times of array sampling chunks
if not os.path.exists(timesdir):
    find_points_of_interest = True

with open(dn+fn1, 'rb') as f:

    if tide is '13': # tide 13 was logged differently - no subsecond, different date format
        clean_lines = ( line.replace(b'R',b'').replace(b':',b' ') for line in f )
        rng = np.genfromtxt(clean_lines,usecols=(2,3,4,5,),delimiter=' ')
    else:
        clean_lines = ( line.replace(b'R',b'').replace(b'-',b' ') for line in f )
        rng = np.genfromtxt(clean_lines,usecols=(0,1,2,3,4,5,6,),delimiter=' ')

if tide is '13':
    range1 = rng[:,3]
    npts = len(range1)
    tmin = build_unixtime(2018, 10, 20, rng[0,0], rng[0,1], rng[0,2])
    tmax = build_unixtime(2018, 10, 20, rng[-1,0], rng[-1,1], rng[-1,2])
    date1 = np.linspace(tmin, tmax, npts)

else:
    range1 = rng[:,6]

    # bilud unix time vector
    date1 = []
    for i in range(0, len(rng[:,0])):
        date1.append(build_unixtime(rng[i,0], rng[i,1], rng[i,2], rng[i,3], rng[i,4], rng[i,5]))


# try focusing on bed data only -- call it the the bed if n [6-12?] preceding points are different by less than x mm [6?]
b_range = 12 # preceding range for use in determing if bed ...
d_thresh = 6

bed_vec = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

for ii in range(b_range, len(range1)):
    for jj in range(1, b_range):
        if abs(range1[ii] - range1[ii-jj]) > d_thresh:
            bval = float('nan')
            break
        bval = np.mean(range1[ii-b_range+1:ii])
    bed_vec.append(bval)

bed_vec = np.array(bed_vec)
date1 = np.array(date1)
Ibed = np.argwhere(~np.isnan(bed_vec))#.item()
bed_good = bed_vec[Ibed]
date_good = date1[Ibed]

raw_bed_level = np.array([date_good, bed_good])
raw_range = np.array([date1, range1])



plt.figure(6)
plt.plot(date1,range1)
plt.plot(date_good,bed_good, '.')
#plt.plot(date1,bed_vec)
plt.xlabel('time [s]')
plt.ylabel('Range [mm]')
# plt.ylim([550, 1150])
# plt.xlim([5000, 7000])
plt.show()

if find_points_of_interest is True:

    plt.figure(7)
    plt.plot(date_good,bed_good, '.')
    plt.xlabel('time [s]')
    plt.ylabel('Range [mm]')
    # plt.ylim([550, 1150])
    # plt.xlim([5000, 7000])
    plt.show()
    POI = plt.ginput(-1, timeout=0, show_clicks=True)

    POI = np.array(POI)

    plt.figure(77)
    plt.plot(date_good,bed_good, '.')
    plt.xlabel('time [s]')
    plt.ylabel('Range [mm]')
    plt.plot(POI[:,0], POI[:,1], 'r.')

    np.save(timesdir, POI)

else:

    POI = np.load(timesdir)


# SMOOTH CHUNKS:

raw_chunks = {}
smoothed_chunks = np.empty((0))
chunk_time = np.empty((0))


for ii in range(0, int(len(POI)/2)):

    start_time = POI[ii*2][0]
    end_time = POI[1 + ii*2][0]

    time_chunk_indices = np.where(np.logical_and(date_good > start_time, date_good < end_time))
    time_chunk = date_good[time_chunk_indices]
    bed_chunk = bed_good[time_chunk_indices]

#    plt.figure(78)
#    plt.plot(date_good,bed_good, '.')
#    plt.xlabel('time [s]')
#    plt.ylabel('Range [mm]')
#    plt.plot(time_chunk,bed_chunk, 'r.')

    # interpolate, then smooth
    f = interpolate.interp1d(time_chunk, bed_chunk)
    time_chunk_reg = np.linspace(np.min(time_chunk), np.max(time_chunk), len(time_chunk)*5)
    bed_chunk_reg = f(time_chunk_reg)

    cutoff = 6 #[mins]
    cutoff_freq = 1/60/cutoff/(1/np.mean(np.diff(time_chunk_reg))/2) # normalized: (1/60/cutoff) / Nyquist
    b, a = signal.butter(2, cutoff_freq)
    bed_chunk_filt = signal.filtfilt(b, a, bed_chunk_reg, padlen=150)

    plt.figure(1000+ii)
#    plt.subplot(211)
    plt.plot(date_good,bed_good, '.')
    plt.xlabel('time [s]')
    plt.ylabel('Range [mm]')
    plt.plot(time_chunk_reg, bed_chunk_reg, 'r-')
#    plt.subplot(212)
    plt.plot(time_chunk_reg, bed_chunk_filt, 'k-')

    raw_bed = np.array([time_chunk, bed_chunk])
#    smooth_bed = np.array([time_chunk_reg, bed_chunk_filt])

    raw_chunks['chunk' + str(int(ii+1))] = raw_bed
#    smoothed_chunks['chunk' + str(int(ii+1))] = smooth_bed
    smoothed_chunks = np.append(smoothed_chunks, bed_chunk_filt)
    chunk_time = np.append(chunk_time, time_chunk_reg)


# SAVE:

smooth_chunks = np.array([np.array(chunk_time), np.array(smoothed_chunks)])

bed_level = {"raw range": raw_range, "raw bed level": raw_bed_level, "bed level chunks": raw_chunks, "smoothed chunks": smooth_chunks}

## save
homechar = "C:\\"
outdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                "range_data", "bed_level", "tide" + tide)

if not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


#np.save(os.path.join(outdir, fn1[:7] + ".npy"), bed_level)





#vvv = np.load(os.path.join(outdir, fn1[:7] + ".npy")).item()
#smth = vvv['smoothed chunks']
#rtime = smth[0]
#sonar_range = smth[1]
#
#plt.figure(2)
#plt.plot(rtime, sonar_range, '.')
