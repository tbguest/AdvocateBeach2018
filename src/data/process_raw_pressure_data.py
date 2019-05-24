#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:14:25 2018

@author: TBG

Script reads in txt file of pressure data exported using Ruskin, as well as
tide times and ranges from gc tides, and uses tide times to separate pressure
time series into tide-demarcated files.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json


def build_unixtime(year, month, day, hour, minute, second):

    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), 0)
    utime = time.mktime(dt.timetuple()) + second

    return utime


def p_abs2atm(unixtime, pressure):

    ''' Here, I'm going to extrct the sub-aerial pressure signal, and interpolate,
    so I have an atmospheric pressure record

    Method:
        - window in 20 min increments (must be less than time scale of atm
        pressure change)
        - compute stdev / window
        - extract where stdev < thresh.
    '''

    import math
    import statistics as stats

    rec_length = len(unixtime)
    win_length = 20*60*4 # 20 mins * 60 s/min * 4 hz
    nchunks = math.floor(rec_length/win_length)

    p = []
    t = []

    for n in range(0, nchunks):

        p_subset = pressure[ n*win_length : (n+1)*win_length ]
        t_subset = unixtime[ n*win_length : (n+1)*win_length ]

        if stats.stdev(p_subset) < 0.01:
            I_mean = int(stats.mean(range(0, win_length)))
            p.append(p_subset[I_mean])
            t.append(t_subset[I_mean])

    p_atm = np.interp(unixtime, t, p)

    return p_atm


def read_tides(tidefile):

    with open(tidefile, 'rb') as f:
        clean_lines = ( line.replace(b'/',b';').replace(b':',b';').replace(b'(m)',b'').replace(b'(ft)',b'') \
                       for line in f )
        tides = np.genfromtxt(clean_lines,usecols=(0,1,2,3,4,5,6,7,),delimiter=';', skip_header=4)


        # convert tide time date stamps to unix time
        tide_time = []
        for n in np.arange(len(tides[:,1])):
            tide_time.append(build_unixtime(tides[n,0], tides[n,1], tides[n,2], \
                                            tides[n,3], tides[n,4], tides[n,5]))

        # array of high and low tide elevations
        height = tides[:,6]

        # index of first high tide
        if height[1] > height[0]:
            first_high = 1
        else:
            first_high = 0

        # high tides only, with associated times
        high_tides = height[first_high::2]
        high_tide_times = tide_time[first_high::2]

        return high_tide_times, high_tides


def read_pressure(fn):

    # function to convert datestring to unix time
    str2date = lambda x: time.mktime(datetime.strptime(x.decode("utf-8"), \
                        '%Y-%m-%d %H:%M:%S.%f').timetuple()) + \
                        datetime.strptime(x.decode("utf-8"), \
                        '%Y-%m-%d %H:%M:%S.%f').microsecond/1e6

    # read in pressure data (this takes a while...)
    # time:
    with open(fn, 'rb') as f:
    #    clean_lines = ( line.replace(b'-',b',').replace(b':',b',').replace(b' ',b',') for line in f )
        unixtime = np.genfromtxt(f,usecols=(0,), delimiter=',', skip_header=1, \
                                 dtype=(str), converters={0: str2date})
        # unixtime = np.genfromtxt(f,usecols=(0,), delimiter=' ', skip_header=0, \
        #                          dtype=(str), converters={0: str2date})

    # temp, pressure, etc
    with open(fn, 'rb') as f:
        # clean_lines = ( line.replace(b'-',b',').replace(b':',b',').replace(b' ',b',') for line in f )
        data = np.genfromtxt(f,usecols=(1,2,3,4,), delimiter=',', skip_header=1, \
                             dtype=float)
        # data = np.genfromtxt(f,usecols=(1,2,3,4,), delimiter=' ', skip_header=0, dtype=float)

    pressure = data[:,1]

    return unixtime, pressure


def utime2yearday(unixtime):

    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday



def main():

    import os

    # for portability
    # homechar = "C:\\"
    homechar = os.path.expanduser("~") # linux

    dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "raw", \
                      "Pressure","051068_20181101_1752")

    # fn = os.path.join(dn, "051068_20181101_1752_eng.txt")
    fn = os.path.join(dn, "051068_20181101_1752_data.txt")

    # tide times
    tidefile = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                            "external", "tide_times.txt")

    unixtime, pressure = read_pressure(fn)
    print('read_pressure: done')
    p_atm = p_abs2atm(unixtime, pressure)
    print('p_abs2atm: done')
    high_tide_times, high_tides = read_tides(tidefile)

    yearday = utime2yearday(unixtime)
    high_tide_yearday = utime2yearday(high_tide_times)

    Irange = int(3.5*60*60*4) # +- indices/tide: 3.5 hrs, 60 min/hr, 60 sec/min, 4 Hz
    # convert full time series to depth
    rho_s = 1030
    g = 9.81
    d = (pressure - p_atm)*10000/(rho_s*g) # 1e5 conversion factor - dbar to Pa

    for t in range(0, len(high_tide_times)):

        Imin = np.argmin(np.abs(yearday - high_tide_yearday[t]))
        if Imin != 0:
    #        Ihigh.append(Imin)
    #        tide_mask[t] = 1.

            # first sampling tide is ind = 6
    #        ind = Ihigh[t]
            ind = Imin
            Ilo = ind - Irange
            Ihi = ind + Irange
            t_tide = yearday[Ilo:Ihi]
            d_tide = d[Ilo:Ihi]

            # .tolist() allows dict to be json serializable
            tide = {"t": t_tide.tolist(), "d": d_tide.tolist(), "high_tide": high_tide_yearday[t].tolist(), "high_tide_elevation": high_tides[t].tolist()}

            fn_npy = "tide" + str(t + 1) + ".npy"
            fn_json = "tide" + str(t + 1) + ".json"

            svdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                                 "interim","pressure")

            if not os.path.exists(svdir):
                os.makedirs(svdir, exist_ok=True)

            np.save(os.path.join(svdir, fn_npy), tide)

            with open(os.path.join(svdir, fn_json), 'w') as fp:
                json.dump(tide, fp)

if __name__ == '__main__':
    main()
