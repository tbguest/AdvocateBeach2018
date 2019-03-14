
# this needs to be compartmentalized. Hopefully this code will become obsolete, when better code is written

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal
import time
from datetime import datetime
# %matplotlib qt5

# utime2yearday(1540311197.75 + 3*60*60)
# utime2yearday(1540311422.75 + 3*60*60)

plt.close("all")
saveFlag = 0

def smooth_pressure(dep):
    ''' accepts a depth signal, p, for a given tide, and returns a smoothed
    curve, rescaled by the beach slope to give mean cross-shore shoreline
    position [m].
    '''

    b, a = signal.butter(2, 0.001)
    mean_shoreline = signal.filtfilt(b, a, dep)/0.12 # assumed beach slope of 0.12

    return mean_shoreline


def utime2yearday(unixtime):

    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday


homechar = "C:\\"

stone_class = 'yellows'

# for tide 19, position 1: scaling = 421.0550841144531 pix/m
tide = "tide19"
position = "position1"
vidspec = "vid_1540304255" # pos1
navg = 10 # 10 frames were averaged
startframe = 1510 + np.int(navg/2)
# this is obtained using "exrtact_scaling_from_swash_images.py"
scaling = 421.0550841144531
# this range must be determined manually be inspecting the timeseries for
# region of little/no cutoff
goodrange = (295.56834780092595, 295.57413483796296)
nbins = 8

# # for tide 19, position 2: scaling = 458.05166513842414 pix/m
# tide = "tide19"
# position = "position2"
# #vidspec = "vid_1540304255" # pos1
# vidspec = "vid_1540307860" # pos2
# navg = 30 # 10 frames were averaged
# startframe = 6801 + np.int(navg/2)
# # this is obtained using "exrtact_scaling_from_swash_images.py"
# scaling = 458.05166513842414
# # this range must be determined manually be inspecting the timeseries for
# # region of little/no cutoff
# goodrange = (295.616, 295.622)
# nbins = 8


# # for tide 19, position 3: scaling = 458.05166513842414 pix/m
# tide = "tide19"
# position = "position3"
# #vidspec = "vid_1540304255" # pos1
# vidspec = "vid_1540307860" # pos2
# navg = 10 # 10 frames were averaged
# startframe = 12751 + np.int(navg/2)
# # this is obtained using "exrtact_scaling_from_swash_images.py"
# scaling = 472.43762017669604
# # this range must be determined manually be inspecting the timeseries for
# # region of little/no cutoff
# goodrange = (295.63423321759257, 295.63683738425925)
# nbins = 8


# # for tide 19, position 2: scaling = 458.05166513842414 pix/m
# tide = "tide19"
# position = "position4"
# vidspec = "vid_1540311466" # pos 4
# navg = 10 # 10 frames were averaged
# startframe = 701 + np.int(navg/2)
# # this is obtained using "exrtact_scaling_from_swash_images.py"
# scaling = 436.65137206616646
# # this range must be determined manually be inspecting the timeseries for
# # region of little/no cutoff
# goodrange = (295.6416811342593, 295.64982060185184)
# nbins = 8


# pressure data for determining mean shoreline
pressurefn = os.path.join(homechar,'Projects','AdvocateBeach2018','data','interim','pressure',tide + '.npy')
p = np.load(pressurefn).item()
tt = p['t']
dep = p['d']
mean_shoreline = smooth_pressure(dep) # not yet aligned with swash

# load swash timestack/timeseries data
tstackdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', 'interim', 'swash', tide, position)
dn1 = os.path.join(tstackdir, 'timeseries', vidspec)
# dn2 = os.path.join(tstackdir, 'timestacks', vidspec)
jnk = np.load(os.path.join(dn1, 'timeseries.npy')).item()
tvec = jnk['timevec']
t_swash = utime2yearday(tvec) + 3*60*60/86400 # convert to UTC
# rescale:
tseries0200 = jnk['line0200']/scaling
tseries0600 = jnk['line0600']/scaling
tseries1000 = jnk['line1000']/scaling

'''
Find time series indices associated with manually identified "goodrange".
Use this time series segment to compute mean, variance, stdevself.

The mean allows the offset between shoreline and swash to be computed. Stdev or
variance can be used to define swash zone width.
'''
Imeantide = np.where(np.logical_and(tt>=goodrange[0], tt<=goodrange[1]))
Imeanswash = np.where(np.logical_and(t_swash>=goodrange[0], t_swash<=goodrange[1]))

meantide = np.mean(mean_shoreline[Imeantide])
meanswash1 = np.mean(tseries0200[Imeanswash])
meanswash2 = np.mean(tseries0600[Imeanswash])
meanswash3 = np.mean(tseries1000[Imeanswash])

varianceswash1 = np.var(signal.detrend(tseries0200[Imeanswash]))
varianceswash2 = np.var(signal.detrend(tseries0600[Imeanswash]))
varianceswash3 = np.var(signal.detrend(tseries1000[Imeanswash]))
varianceswash = np.mean([varianceswash1, varianceswash2, varianceswash3])

stdswash1 = np.std(signal.detrend(tseries0200[Imeanswash]))
stdswash2 = np.std(signal.detrend(tseries0600[Imeanswash]))
stdswash3 = np.std(signal.detrend(tseries1000[Imeanswash]))
stdswash = np.mean([stdswash1, stdswash2, stdswash3])

offset1 = meantide - meanswash1
offset2 = meantide - meanswash2
offset3 = meantide - meanswash3
offset = np.mean([offset1, offset2, offset3])

######### save swash timeseries with proper time vector and scaled coordinates
dn_saveswash = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', \
            'data', 'processed', 'swash', tide, 'camera_' + position)

swash_timeseries = {'tvec': t_swash, '0200': tseries0200 + offset, \
            '0600': tseries0600 + offset, '1000': tseries1000 + offset, \
            'stdev': stdswash}

if not os.path.exists(dn_saveswash):
    try:
        os.makedirs(dn_saveswash)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

np.save(os.path.join(dn_saveswash, 'swash_timeseries.npy'), swash_timeseries)
##########################################################################

# plot for verification
plt.figure(1)
plt.plot(t_swash, tseries0200 + offset)
plt.plot(tt,mean_shoreline)
plt.plot(tt,mean_shoreline + 2*stdswash, 'r')
plt.plot(tt,mean_shoreline - 2*stdswash, 'r')
plt.xlabel('yearday')
plt.ylabel('cross-shore distance [m]')
# plt.get_xaxis().get_major_formatter().set_useOffset(False)


#########################################

# load trajectory data
trajdir = os.path.join(homechar, 'Projects','AdvocateBeach2018','data','interim','cobble_tracking',tide, position,vidspec)

if stone_class == 'yellows':
    stone_clr = glob.glob(os.path.join(trajdir, 'yellow*.npy'))
elif stone_class == 'reds':
    stone_clr = glob.glob(os.path.join(trajdir, 'red*.npy'))
elif stone_class == 'blues':
    stone_clr = glob.glob(os.path.join(trajdir, 'blue*.npy'))
else:
    raise Exception('Invalid stone colour. Try again.')

posixtime0 = float(vidspec[-10:])

t_cobble = {}
pixloc = {}
for stone in sorted(stone_clr):
    jnk = np.load(stone).item()
    count = jnk['count']
    t_cobble[int(stone[-6:-4])] = utime2yearday(posixtime0) + 3*60*60/86400 + (np.array(count) + startframe)/4/86400
    # t_cobble[int(stone[-6:-4])] = utime2yearday(posixtime0) + (np.array(count) + 6801.0)/4/86400
    pixloc[int(stone[-6:-4])] = jnk['position']

# plots the shoreline adjusted x coordinate of stones at each timestep vs the
# subsequent change in x-coordinate. Looking for trends in transport direction
# and magnitude based on position in swash zone
dx_sum = []
dx_sum_correction = []
dy_sum = []

abs_dx_sum = []
abs_dy_sum = []

# images were rescaled to 1000 during tracking step:
rescale_pix_x = 1640/1000
# rescale_pix_y = 1232/1000

# initailize bins
bins = {}
bins_correction = {}
residence_time_bins = {}
# for x in np.arange(nbins):
#     bins[x] = None
for x in np.arange(nbins):
    residence_time_bins[x] = 0

transport_thresh = 0.01 # 1cm

# initialize plots
fig1, ax1 = plt.subplots(nrows=1, ncols=1, num="scatter")
fig2, ax2 = plt.subplots(nrows=1, ncols=1, num="trajectory timeseries")
fig3, ax3 = plt.subplots(nrows=1, ncols=1, num="cum. sum")
fig4, ax4 = plt.subplots(nrows=1, ncols=1, num="cum. abs. sum")

# initailize dict to be saved in final form
yellow_stones = {}

for i in range(1, len(t_cobble)+1):

    shoreline_correction = []

    foo = np.array(pixloc[i])
    xpix = foo[:,0]
    ypix = foo[:,1]

    trsh = xpix/1000*nbins
    trsh2 = trsh.astype(int)

    # (trsh2 == 1).sum()

    # bin cobble position x values for a later computation of residence time
    for bn in np.arange(nbins):
        residence_time_bins[bn] += (trsh2 == bn).sum()

    # np.sum(1 for i in np.arange(nbins) if i % 4 == 3)

    # for m in np.arange(len(xpix)):
    #     xbinsum = np.int(xpix[m]/1000*nbins)
    #     residence_time_bins[xbinsum] += 1
    #     if xbinsum not in residence_time_bins:
    #         residence_time_bins[xbinsum] = 1
    #     else:
    #         residence_time_bins.setdefault(xbinsum, []).append(1) # worked at this one for a while...
    #

    # plt.plot(xpix/1000*nbins)

    for j in range(len(pixloc[i])):

        Imeanshore = np.argmin(np.abs(t_cobble[i][j] - tt))
        shoreline_correction.append(mean_shoreline[Imeanshore])

    # coordinates / 1000
    xcob = np.array(xpix)*rescale_pix_x/scaling + offset
    xcob_correction = np.array(xpix)*rescale_pix_x/scaling + offset - np.array(shoreline_correction)
    ycob = np.array(ypix)*rescale_pix_x/scaling

    dxcob = np.diff(xcob)
    dxcob_correction = np.diff(xcob_correction)
    dycob = np.diff(ycob)
    dx_sum.append(np.sum(dxcob))
    dx_sum_correction.append(np.sum(dxcob_correction))
    dy_sum.append(np.sum(dycob))

    abs_dx_sum.append(np.sum(np.abs(dxcob)))
    abs_dy_sum.append(np.sum(np.abs(dycob)))

    # find indices of transport > thresh (to leave out changes due to reinitialization of tracker)
    Itransport = np.where(np.abs(dxcob) > transport_thresh)
    for k in range(len(Itransport[0])):

        # binning in image-centric reference frame
        binx = np.int(xpix[Itransport[0][k]]/1000*nbins) # i think this gives the right starting point...
        # append to dict bin
        if binx not in bins:
            bins[binx] = [dxcob[Itransport[0][k]]]
        else:
            bins.setdefault(binx, []).append(dxcob[Itransport[0][k]]) # worked at this one for a while...

        # for binning in swash-centric reference frame:
        binx_correction = np.int((xcob_correction[Itransport[0][k]] + stdswash)/(2*stdswash)*(nbins)+1)
        # binx_correction = np.int(xcob_correction[Itransport[0][k]]/1000*nbins)
        if binx_correction not in bins_correction:
            bins_correction[binx_correction] = [dxcob_correction[Itransport[0][k]]] # [-0.01370234132576087]
        else:
            bins_correction.setdefault(binx_correction, []).append(dxcob_correction[Itransport[0][k]]) # worked at this one for a while...

    # update plots
    ax1.plot(xcob_correction[:-1], dxcob_correction, 'y.')
    ax2.plot(t_cobble[i], xcob)
    ax3.plot(t_cobble[i][:-1], np.cumsum(dxcob))
    ax4.plot(t_cobble[i][:-1], np.cumsum(np.abs(dxcob)))


    ######## create (t, x, y) dict of cobble data for saving #############
    yellow_stones[i] = {'t': t_cobble[i], 'x': xcob, 'y': ycob, \
                'x_pixel': xpix, 'y_pixel': ypix}
    ######################################################################


######## save dict of cobble data ########################################
dn_savecobble = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', \
            'data', 'processed', 'cobble_tracking', tide, 'camera_' + position)

if not os.path.exists(dn_savecobble):
    try:
        os.makedirs(dn_savecobble)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

np.save(os.path.join(dn_savecobble, stone_class + '.npy'), yellow_stones)
##########################################################################

# size class statistics
# I can compute these for all events and plot (mean +/- std), giving context for
# differences via (e.g.) stage of tide , background composition, forcing
mean_xtransport = np.mean(np.array(dx_sum))
mean_ytransport = np.mean(np.array(dy_sum))
std_xtransport = np.std(np.array(dx_sum))
std_ytransport = np.std(np.array(dy_sum))

abs_xtransport = np.mean(abs_dx_sum)
abs_ytransport = np.mean(abs_dy_sum)
std_abs_xtransport = np.std(abs_dx_sum)
std_abs_ytransport = np.std(abs_dy_sum)

# compute residence time
restime = [0,0,0,0,0,0,0,0]
restime = [residence_time_bins[i]/4/len(t_cobble) for i in np.arange(len(residence_time_bins))]

# plot binned data
binmean = np.zeros(nbins)
binstd = np.zeros(nbins)
bincount = np.zeros(nbins)
imgbins = np.arange(nbins)
for n in range(nbins):
    if n in bins:
        binmean[n] = np.mean(bins[n])
        binstd[n] = np.std(bins[n])
        bincount[n] = len(bins[n])

binmean_swash = np.zeros(nbins)
binstd_swash = np.zeros(nbins)
bincount_swash = np.zeros(nbins)
imgbins_swash = np.arange(nbins)
for n in range(nbins):
    if n in bins_correction:
        binmean_swash[n] = np.mean(bins_correction[n])
        binstd_swash[n] = np.std(bins_correction[n])
        bincount_swash[n] = len(bins_correction[n])

ax1.set_ylabel('transport distance [m]')
ax1.set_xlabel('starting point rel. to mean swash [m]')

ax2.set_ylabel('cross-shore distance [m]')
ax2.set_xlabel('yearday')

ax3.set_ylabel('cum. sum of cross-shore transport [m]')
ax3.set_xlabel('yearday')

ax4.set_ylabel('cum. sum of abs. cross-shore transport [m]')
ax4.set_xlabel('yearday')

fig101, (ax201, ax101) = plt.subplots(nrows=2, ncols=1, num="binned mean transport", gridspec_kw = {'height_ratios':[1, 3]})
ax101.plot(imgbins, binmean, 'ko')
ax101.plot([imgbins, imgbins], [binmean-binstd, binmean+binstd], 'k-')
ax101.plot(np.linspace(0, nbins-1, 50), np.zeros(50), 'k--')
ax101.set_ylabel('cross-shore distance [m]')
ax101.set_xlabel('bin')
ax201.plot(imgbins, bincount, 'ko')
ax201.set_ylabel('count')
# ax201.set_xlabel('bin')
ax201.set_ylim(0, np.max(bincount)+0.1*(np.max(bincount)))
fig101.tight_layout()

fig102, (ax102) = plt.subplots(nrows=1, ncols=1, num="binned mean transport: swash")
ax102.plot(imgbins_swash, binmean_swash, 'ko')
ax102.plot([imgbins_swash, imgbins_swash], [binmean_swash-binstd_swash, binmean_swash+binstd_swash], 'k-')
ax102.plot(np.linspace(0, nbins, 50), np.zeros(50), 'k--')
ax102.set_ylabel('cross-shore distance [m]')
ax102.set_xlabel('bin')

fig103, (ax103) = plt.subplots(nrows=1, ncols=1, num="binned residence time")
ax103.plot(imgbins_swash, restime, 'ko')
ax103.set_ylabel('residence time per stone [s]')
ax103.set_xlabel('cross-shore bin')

fig104, (ax104) = plt.subplots(nrows=1, ncols=1, num="binned transport/s/stone")
ax104.plot(imgbins_swash, np.abs(binmean_swash/restime/bincount), 'ko')
ax104.set_ylabel('net transport/s/stone [s]')
ax104.set_xlabel('cross-shore bin')


# save figures #################################################################
savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','cobble_transport',tide,position)
if not os.path.exists(savedn):
    try:
        os.makedirs(savedn)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


if saveFlag == 1:

    # fig1.savefig(os.path.join(savedn, 'transport_histogram.png'), dpi=600, transparent=True)
    # fig1.savefig(os.path.join(savedn, 'transport_histogram.pdf'), dpi=None, transparent=True)

    fig2.savefig(os.path.join(savedn, 'transport_timeseries.png'), dpi=600, transparent=True)
    fig2.savefig(os.path.join(savedn, 'transport_timeseries.pdf'), dpi=None, transparent=True)

    # fig3.savefig(os.path.join(savedn, 'transport_histogram.png'), dpi=600, transparent=True)
    # fig3.savefig(os.path.join(savedn, 'transport_histogram.pdf'), dpi=None, transparent=True)

    # fig4.savefig(os.path.join(savedn, 'transport_histogram.png'), dpi=600, transparent=True)
    # fig4.savefig(os.path.join(savedn, 'transport_histogram.pdf'), dpi=None, transparent=True)

    fig101.savefig(os.path.join(savedn, 'transport_histogram.png'), dpi=600, transparent=True)
    fig101.savefig(os.path.join(savedn, 'transport_histogram.pdf'), dpi=None, transparent=True)

    fig102.savefig(os.path.join(savedn, 'transport_histogram_swash.png'), dpi=600, transparent=True)
    fig102.savefig(os.path.join(savedn, 'transport_histogram_swash.pdf'), dpi=None, transparent=True)

    fig103.savefig(os.path.join(savedn, 'residence_time_histogram.png'), dpi=600, transparent=True)
    fig103.savefig(os.path.join(savedn, 'residence_time_histogram.pdf'), dpi=None, transparent=True)
