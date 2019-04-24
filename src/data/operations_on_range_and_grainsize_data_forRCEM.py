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

%matplotlib qt5

# change default font size
plt.rcParams.update({'font.size': 12})

def utime2yearday(unixtime):

    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday


tide = '27'
chunk = '2'

post = 1.7 # survey post height

homechar = "C:\\"

gsdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'grainsize', 'pi_array', 'tide' + tide)

rngdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'bed_level', 'tide' + tide)

# pi locations
pifile = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'GPS', 'array', 'tide' + tide, \
                           'array_position' + chunk + '.npy')


aaa = np.load(os.path.join(gsdir, 'chunk' + chunk + '.npy')).item()
bbb = np.load(os.path.join(rngdir, 'chunk' + chunk + '.npy')).item()

pilocs = np.load(pifile).item()

f0 = 0.25
iters = 2

ff = len(aaa['pi71']['tvec'])
tmin = aaa['pi71']['tvec'][0]
tmax = aaa['pi71']['tvec'][-1]

t71 = aaa['pi71']['tvec']
# t72 = aaa['pi72']['tvec']
t73 = aaa['pi73']['tvec']
t74 = aaa['pi74']['tvec']



if tide == '27':

#    mgs_fit1, yout, wout = loess_1d(aaa['pi73']['tvec'], aaa['pi73']['mgs'], frac=f0, degree=1, rotate=False)
    #
    # z_fit1 = lowess.lowess(utime2yearday(bbb['pi71']['tvec']), pilocs['z'][0] + post - bbb['pi71']['range']/1000, f=f0, iter=iters)

    mgs_fit1 = lowess.lowess(aaa['pi71']['tvec'], aaa['pi71']['mgs'], f=f0, iter=iters)
    #mgs_fit2 = lowess.lowess(aaa['pi72']['tvec'], aaa['pi72']['mgs'], f=f0, iter=iters)
    # mgs_fit3 = lowess.lowess(aaa['pi73']['tvec'], aaa['pi73']['mgs'], f=f0, iter=iters)
    # mgs_fit4 = lowess.lowess(aaa['pi74']['tvec'], aaa['pi74']['mgs'], f=f0, iter=iters)

    sort_fit1 = lowess.lowess(aaa['pi71']['tvec'], aaa['pi71']['sort'], f=f0, iter=iters)
    #sort_fit2 = lowess.lowess(aaa['pi72']['tvec'], aaa['pi72']['sort'], f=f0, iter=iters)
    # sort_fit3 = lowess.lowess(aaa['pi73']['tvec'], aaa['pi73']['sort'], f=f0, iter=iters)
    # sort_fit4 = lowess.lowess(aaa['pi74']['tvec'], aaa['pi74']['sort'], f=f0, iter=iters)


    fig1 = plt.figure(1, figsize=(5.5,4.5))

    ax1 = fig1.add_subplot(311)
    ax1.plot(utime2yearday(bbb['pi71']['tvec']), pilocs['z'][0] + post - bbb['pi71']['range']/1000, '.k')
    # ax1.plot(utime2yearday(bbb['pi71']['tvec']), z_fit1, 'k', Linewidth=2)
    #ax1.plot(bbb['pi72']['tvec'], bbb['pi72']['range'], '.C1')
    # ax1.plot(utime2yearday(bbb['pi73']['tvec']), pilocs['z'][0] + post - bbb['pi73']['range']/1000, '.C2')
    # ax1.plot(utime2yearday(bbb['pi74']['tvec']), pilocs['z'][0] + post - bbb['pi74']['range']/1000, '.C3')
    lms = ax1.get_ylim()
    ax1.plot([299.681, 299.681], [0, 5], 'k--')
    ax1.set_ylim([lms[0], lms[1]])
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('elevation [m]')
    ax1.xaxis.set_major_formatter(plt.NullFormatter())


    ax2 = fig1.add_subplot(312)
    ax2.plot(utime2yearday(aaa['pi71']['tvec']), aaa['pi71']['mgs'], '.k')
    # #ax2.plot(aaa['pi72']['tvec'], aaa['pi72']['mgs'], '.C1')
    # ax2.plot(utime2yearday(aaa['pi73']['tvec']), aaa['pi73']['mgs'], '.C2')
    # ax2.plot(utime2yearday(aaa['pi74']['tvec']), aaa['pi74']['mgs'], '.C3')
    ax2.plot(utime2yearday(aaa['pi71']['tvec']), mgs_fit1, 'k', Linewidth=2)
    # #ax2.plot(aaa['pi72']['tvec'], mgs_fit2, 'C1', Linewidth=2)
    # ax2.plot(utime2yearday(aaa['pi73']['tvec']), mgs_fit3, 'C2', Linewidth=2)
    # ax2.plot(utime2yearday(aaa['pi74']['tvec']), mgs_fit4, 'C3', Linewidth=2)
    lms = ax2.get_ylim()
    ax2.plot([299.681, 299.681], [0, 50], 'k--')
    ax2.set_ylim([lms[0], lms[1]])
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('grain size [mm]')
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    # ax2.set_xlabel('yearday')
    # fig1.tight_layout()




    ax3 = fig1.add_subplot(313)
    ax3.plot(utime2yearday(aaa['pi71']['tvec']), aaa['pi71']['sort'], '.k')
    # #ax3.plot(aaa['pi72']['tvec'], aaa['pi72']['sort'], '.C1')
    # ax3.plot(utime2yearday(aaa['pi73']['tvec']), aaa['pi73']['sort'], '.C2')
    # ax3.plot(utime2yearday(aaa['pi74']['tvec']), aaa['pi74']['sort'], '.C3')
    ax3.plot(utime2yearday(aaa['pi71']['tvec']), sort_fit1, 'k', Linewidth=2)
    lms = ax3.get_ylim()
    ax3.plot([299.681, 299.681], [0, 50], 'k--')
    ax3.set_ylim([lms[0], lms[1]])
    # #ax3.plot(aaa['pi72']['tvec'], sort_fit2, 'C1', Linewidth=2)
    # ax3.plot(utime2yearday(aaa['pi73']['tvec']), sort_fit3, 'C2', Linewidth=2)
    # ax3.plot(utime2yearday(aaa['pi74']['tvec']), sort_fit4, 'C3', Linewidth=2)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.set_ylabel('sorting [mm]')
    ax3.set_xlabel('yearday')
    fig1.tight_layout()


    fig2 = plt.figure(2)

    ax1 = fig2.add_subplot(311)
    ax1.plot(bbb['pi71']['tvec'], pilocs['z'][0] + post - bbb['pi71']['range']/1000, '.C0')
    # ax1.plot(bbb['pi73']['tvec'], pilocs['z'][0] + post - bbb['pi73']['range']/1000, '.C2')
    # ax1.plot(bbb['pi74']['tvec'], pilocs['z'][0] + post - bbb['pi74']['range']/1000, '.C3')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('elevation [m]')

    zvec = np.zeros(aaa['pi74']['tvec'].shape)

    ax2 = fig2.add_subplot(312)
    ax2.plot(aaa['pi71']['tvec'][1:], np.diff(mgs_fit1)/np.diff(aaa['pi71']['tvec']), 'C0', Linewidth=2)
    # ax2.plot(aaa['pi73']['tvec'][1:], np.diff(mgs_fit3)/np.diff(aaa['pi73']['tvec']), 'C2', Linewidth=2)
    # ax2.plot(aaa['pi74']['tvec'][1:], np.diff(mgs_fit4)/np.diff(aaa['pi74']['tvec']), 'C3', Linewidth=2)
    ax2.plot(aaa['pi74']['tvec'], zvec, 'k--')
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('d(mgs)/dt [mm/s]')
    ax2.set_xlabel('time [s]')


else:

    mgs_fit1 = lowess.lowess(aaa['pi71']['tvec'], aaa['pi71']['mgs'], f=f0, iter=iters)
    mgs_fit2 = lowess.lowess(aaa['pi72']['tvec'], aaa['pi72']['mgs'], f=f0, iter=iters)
    mgs_fit3 = lowess.lowess(aaa['pi73']['tvec'], aaa['pi73']['mgs'], f=f0, iter=iters)
    mgs_fit4 = lowess.lowess(aaa['pi74']['tvec'], aaa['pi74']['mgs'], f=f0, iter=iters)

    sort_fit1 = lowess.lowess(aaa['pi71']['tvec'], aaa['pi71']['sort'], f=f0, iter=iters)
    sort_fit2 = lowess.lowess(aaa['pi72']['tvec'], aaa['pi72']['sort'], f=f0, iter=iters)
    sort_fit3 = lowess.lowess(aaa['pi73']['tvec'], aaa['pi73']['sort'], f=f0, iter=iters)
    sort_fit4 = lowess.lowess(aaa['pi74']['tvec'], aaa['pi74']['sort'], f=f0, iter=iters)


    fig1 = plt.figure(1, figsize=(8,6))

    ax1 = fig1.add_subplot(311)
    ax1.plot(utime2yearday(bbb['pi71']['tvec']), pilocs['z'][0] + post - bbb['pi71']['range']/1000, '.C0')
    ax1.plot(utime2yearday(bbb['pi72']['tvec']), pilocs['z'][1] + post - bbb['pi72']['range']/1000, '.C1')
    ax1.plot(utime2yearday(bbb['pi73']['tvec']), pilocs['z'][2] + post - bbb['pi73']['range']/1000, '.C2')
    ax1.plot(utime2yearday(bbb['pi74']['tvec']), pilocs['z'][3] + post - bbb['pi74']['range']/1000, '.C3')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('elevation [m]')
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax2 = fig1.add_subplot(312)
    ax2.plot(utime2yearday(aaa['pi71']['tvec']), aaa['pi71']['mgs'], '.C0')
    ax2.plot(utime2yearday(aaa['pi72']['tvec']), aaa['pi72']['mgs'], '.C1')
    ax2.plot(utime2yearday(aaa['pi73']['tvec']), aaa['pi73']['mgs'], '.C2')
    ax2.plot(utime2yearday(aaa['pi74']['tvec']), aaa['pi74']['mgs'], '.C3')
    ax2.plot(utime2yearday(aaa['pi71']['tvec']), mgs_fit1, 'C0', Linewidth=2)
    ax2.plot(utime2yearday(aaa['pi72']['tvec']), mgs_fit2, 'C1', Linewidth=2)
    ax2.plot(utime2yearday(aaa['pi73']['tvec']), mgs_fit3, 'C2', Linewidth=2)
    ax2.plot(utime2yearday(aaa['pi74']['tvec']), mgs_fit4, 'C3', Linewidth=2)
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('mean grain size [mm]')
    # ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())

    ax3 = fig1.add_subplot(313)
    ax3.plot(utime2yearday(aaa['pi71']['tvec']), aaa['pi71']['sort'], '.C0')
    ax3.plot(utime2yearday(aaa['pi72']['tvec']), aaa['pi72']['sort'], '.C1')
    ax3.plot(utime2yearday(aaa['pi73']['tvec']), aaa['pi73']['sort'], '.C2')
    ax3.plot(utime2yearday(aaa['pi74']['tvec']), aaa['pi74']['sort'], '.C3')
    ax3.plot(utime2yearday(aaa['pi71']['tvec']), sort_fit1, 'C0', Linewidth=2)
    ax3.plot(utime2yearday(aaa['pi72']['tvec']), sort_fit2, 'C1', Linewidth=2)
    ax3.plot(utime2yearday(aaa['pi73']['tvec']), sort_fit3, 'C2', Linewidth=2)
    ax3.plot(utime2yearday(aaa['pi74']['tvec']), sort_fit4, 'C3', Linewidth=2)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.set_ylabel('sorting [mm]')
    ax3.set_xlabel('yearday')



    fig2 = plt.figure(2)

    ax1 = fig2.add_subplot(311)
    ax1.plot(bbb['pi71']['tvec'], pilocs['z'][0] + post - bbb['pi71']['range']/1000, '.C0')
    ax1.plot(bbb['pi72']['tvec'], pilocs['z'][0] + post - bbb['pi72']['range']/1000, '.C1')
    ax1.plot(bbb['pi73']['tvec'], pilocs['z'][0] + post - bbb['pi73']['range']/1000, '.C2')
    ax1.plot(bbb['pi74']['tvec'], pilocs['z'][0] + post - bbb['pi74']['range']/1000, '.C3')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('elevation [m]')

    zvec = np.zeros(aaa['pi74']['tvec'].shape)

    ax2 = fig2.add_subplot(312)
    ax2.plot([np.min(aaa['pi74']['tvec']), np.max(aaa['pi74']['tvec'])], [0., 0.])
    ax2.plot(aaa['pi71']['tvec'][1:], np.diff(mgs_fit1)/np.diff(aaa['pi71']['tvec']), 'C0', Linewidth=2)
    ax2.plot(aaa['pi72']['tvec'][1:], np.diff(mgs_fit2)/np.diff(aaa['pi72']['tvec']), 'C1', Linewidth=2)
    ax2.plot(aaa['pi73']['tvec'][1:], np.diff(mgs_fit3)/np.diff(aaa['pi73']['tvec']), 'C2', Linewidth=2)
    ax2.plot(aaa['pi74']['tvec'][1:], np.diff(mgs_fit4)/np.diff(aaa['pi74']['tvec']), 'C3', Linewidth=2)
    ax2.plot(aaa['pi74']['tvec'], zvec, 'k--')
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('d(mgs)/dt [mm/s]')
    ax2.set_xlabel('time [s]')

#    ax3 = fig.add_subplot(313)
#    ax3.plot(aaa['pi71']['tvec'], sort_fit1, 'C0', Linewidth=2)
#    ax3.plot(aaa['pi72']['tvec'], sort_fit2, 'C1', Linewidth=2)
#    ax3.plot(aaa['pi73']['tvec'], sort_fit3, 'C2', Linewidth=2)
#    ax3.plot(aaa['pi74']['tvec'], sort_fit4, 'C3', Linewidth=2)
#    ax3.autoscale(enable=True, axis='x', tight=True)



saveFlag = 0
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD',tide)
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig1.savefig(os.path.join(savedn, 'array_timeseriesRCEM2.jpg'), dpi=750, transparent=True)
    fig1.savefig(os.path.join(savedn, 'array_timeseriesRCEM2.pdf'), dpi=None, transparent=True)

    # fig2.savefig(os.path.join(savedn, 'transport_stats.png'), dpi=1000, transparent=True)
    # fig2.savefig(os.path.join(savedn, 'transport_stats.pdf'), dpi=None, transparent=True)
