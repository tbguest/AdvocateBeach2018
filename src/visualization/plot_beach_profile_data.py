#!/usr/bin/env python3
"""
Created on Thu Nov 22 15:47:31 2018

@author: Owner
"""

# %reset

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import stats
from scipy.signal import correlate
import json
from scipy.stats.stats import pearsonr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.colors as mcolors
import matplotlib.colors

# %matplotlib qt5

plt.close('all')

# change default font size
plt.rcParams.update({'font.size': 10})


def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(len(x)-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def save_figures(dn, fn, fig):
    ''' Saves png and pdf of figure.

    INPUTS
    dn: save directory. will be created if doesn't exist
    fn: file name WITHOUT extension
    fig: figure handle
    '''

    if not os.path.exists(dn):
        try:
            os.makedirs(dn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig.savefig(os.path.join(dn, fn + '.png'), dpi=1000, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.pdf'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.eps'), dpi=None, transparent=True)
    fig.savefig(os.path.join(dn, fn + '.jpg'), dpi=1000, transparent=True)


def find_fit_values(ys_orig, xs_orig, ys_line, xs_line):
    ys_fit = [ys_line[np.argmin(np.abs(xi - xs_line))] for xi in xs_orig]

    return np.array(ys_fit)

def coefficient_of_determination(ys_orig, xs_orig, ys_line, xs_line):
    ys_fit = find_fit_values(ys_orig, xs_orig, ys_line, xs_line)
    ssres = np.sum((np.array(ys_orig) - ys_fit)**2)
    sstot = np.sum((np.array(ys_orig) - np.mean(ys_orig))**2)

    return 1 - (ssres/sstot)


def linear_regression(ys_orig, xs_orig):
    lfit = np.polyfit(xs_orig, ys_orig, 1)
    tmprange = np.max(xs_orig) - np.min(xs_orig)
    tmpx = np.linspace(np.min(xs_orig) - 0.05*tmprange, np.max(xs_orig) + 0.05*tmprange, 1000) # 5% buffer on new x vector
    r = coefficient_of_determination(ys_orig, xs_orig, np.polyval(lfit, tmpx), tmpx)

    return lfit, r, tmpx


def smooth_profile(profiles):

    from src.data.regresstools import lowess

    # for troubleshooting:
    # profiles = mgs_line[:maxreal,:]
    # ii=0

    f0 = 3/len(profiles)
    iters = 1

    smooth_profiles = np.zeros(np.shape(profiles))

    for ii in range(len(profiles[:,1])):
        profile = profiles[ii,:]

        smooth_profile = lowess.lowess(np.arange(len(profile)), profile, f=f0, iter=iters)
        smooth_profiles[ii,:] = smooth_profile

    return smooth_profiles


def lin_fit_slope(z_line):

    # for troubleshooting:
    # ii=0

    profiles = z_line#[:maxreal,:]

    a = []

    # append list of non nan values for each profile
    for ii in range(len(profiles[1,:])):
        a.append(len(profiles[~np.isnan(profiles[:,ii])]))

    # index of longest profile
    Ia = np.argmax(a)
    # longest profile, without nans, from station 10 seaward
    longest_profile = profiles[:,Ia][~np.isnan(profiles[:,Ia])]
    lwr_slope = longest_profile[10:]

    # fit line to lower beach
    beach_fit_coeffs = np.polyfit(np.arange(10,len(lwr_slope)+10), lwr_slope, 1)
    beach_fit = np.polyval(beach_fit_coeffs, np.arange(50))

    plt.figure(888)
    plt.plot(beach_fit[:len(longest_profile)])
    plt.plot(longest_profile)

    lindiff_profiles = np.zeros(np.shape(profiles[:maxreal,:])) #[~np.isnan(profiles)]

    for ii in range(len(profiles[1,:])):
        profile = profiles[:,ii][~np.isnan(profiles[:,ii])]

        lindiff = profile[:maxreal] - beach_fit[:len(profile[:maxreal])]
        lindiff_profiles[:,ii] = lindiff

    return lindiff_profiles



# def main():
saveFlag = 0
saveCorr = 0

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux
drivechar = '/media/tristan2/Advocate2018_backup2'


figsdn = os.path.join(homechar,'Projects','AdvocateBeach2018',\
'reports','figures')

# grid_spec = "cross_shore"
grid_spec = "longshore2"
# grid_spec = "longshore1"

hwl = [-21,-9,-15,-15,-18,-15,-15,-15,-18,-21,-18,-18,-18,-18]
# hwl = [-9,-15,-15,-18,-15,-15,-15,-18,-21,-18,-18,-18,-18]


if grid_spec == 'cross_shore':
    start_tide = 13
    # start_tide = 14
    # hwl = hwl[1:]
elif grid_spec == 'longshore1':
    start_tide = 15
    hwl = hwl[2:]
else:
    start_tide = 14
    hwl = hwl[1:]

# if grid_spec == 'cross_shore':
#     start_tide = 13
#     # start_tide = 14
#     # hwl = hwl[1:]
# elif grid_spec == 'longshore1':
#     start_tide = 13
# else:
#     start_tide = 14
#     hwl = hwl[1:]

tide_range = range(start_tide, 28)
tide_axis = np.arange(start_tide+1,28) # for plotting later


counter = 0

# guess at cross-shore profile length
npts = 50

sum_dz_line = np.zeros((npts,len(tide_range)-1))
dz_line = np.zeros((npts,len(tide_range)-1))
z_line = np.zeros((npts,len(tide_range)-1))
sum_dz = np.zeros((npts,))
net_dz = []

sum_dmgs_line = np.zeros((npts,len(tide_range)-1))
dmgs_line = np.zeros((npts,len(tide_range)-1))
mgs_line = np.zeros((npts,len(tide_range)-1))
sum_dmgs = np.zeros((npts,))

sum_dsort_line = np.zeros((npts,len(tide_range)-1))
dsort_line = np.zeros((npts,len(tide_range)-1))
sort_line = np.zeros((npts,len(tide_range)-1))
sum_dsort = np.zeros((npts,))

# initialize
corrcoeffs_dz_mgs= []
corrcoeffs_dz_dmgs= []
corrcoeffs_dz_sort= []
corrcoeffs_dz_dsort= []
corrcoeffs_dz_last_mgs = []
corrcoeffs_dz_last_sort = []
corrcoeffs_dz_last_dz = []

sig_dz_mgs = []
sig_dz_dmgs = []
sig_dz_last_mgs = []
sig_dz_last_dz = []


temporal_dz_mgs = []
temporal_dz_dmgs = []

all_dz = []
all_mgs = []
all_dmgs = []
all_last_mgs = []
all_last_dz = []

Hs = []
Tp = []
steepness = []
iribarren = []
wave_energy = []
wave_energy_wind = []
wave_energy_swell = []
maxdepth = []


fig01, ax01 = plt.subplots(nrows=4,ncols=1, num='profiles', sharex=True)

fig006, ax006 = plt.subplots(3,3, figsize=(9,9), num='spatial correlation, tide-tide')

for ii in tide_range:

    tide = "tide" + str(ii)

    # gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
    #                       "grainsize", "beach_surveys", tide, grid_spec + ".json")

    # gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
    #                       "grainsize", "beach_surveys_reprocessed", tide, grid_spec + ".json")
    gsizefn = os.path.join(drivechar, "data", "processed", \
                          "grainsize", "beach_surveys_reprocessed_x08", tide, grid_spec + ".json")

    gpsfn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                          "GPS", "by_tide", tide, grid_spec + ".json")

    wavefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                  "pressure", "wavestats", tide + ".json")

    # wave data
    if not os.path.exists(wavefn):
        continue

    # GPS data
    if not os.path.exists(gpsfn):
        continue

    # grainsize data
    if not os.path.exists(gsizefn):
        continue

    # bar = np.load(wavefn, allow_pickle=True).item()
    with open(wavefn, 'r') as fpp:
        bar = json.load(fpp)
    Hs.append(np.mean(np.array(bar["Hs"])))
    Tp.append(np.mean(np.array(bar["Tp"])))
    steepness.append(np.mean(np.array(bar["steepness"])))
    iribarren.append(np.mean(np.array(bar["Iribarren"])))
    wave_energy.append(np.mean(np.array(bar["wave_energy"])))
    wave_energy_wind.append(np.mean(np.array(bar["wave_energy_wind"])))
    wave_energy_swell.append(np.mean(np.array(bar["wave_energy_swell"])))
    maxdepth.append(np.max(bar["depth"]))

    # sediment data
    # jnk = np.load(gsizefn, allow_pickle=True).item()
    with open(gsizefn, 'r') as fpp:
        jnk = json.load(fpp)

    mgs0 = np.array(jnk['mean_grain_size'])
    sort0 = np.array(jnk['sorting'])

    mgs = np.pad(mgs0, (0, npts-len(mgs0)), 'constant', constant_values=(np.nan,np.nan))
    sort = np.pad(sort0, (0, npts-len(sort0)), 'constant', constant_values=(np.nan,np.nan))

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


    # for first iteration
    if counter == 0:

        last_z = z
        dz = z - last_z
        last_dz = dz

        last_mgs = mgs
        last_sort = sort
        dmgs = mgs - last_mgs
        dsort = sort - last_sort

        # dz_for_time_corr = dz[0]

        plt.figure(num='base profile')
        plt.plot(y,z)
        plt.ylabel('z [m]')
        plt.xlabel('y [m]')

    else:

        dz = z - last_z
#        last_z = z

        last_dz0 = last_dz

        last_mgs0 = last_mgs
        last_sort0 = last_sort

        dmgs = mgs - last_mgs
        dsort = sort - last_sort

        # update
        last_z = z
        last_dz = dz
        last_mgs = mgs
        last_sort = sort

        # populate space-time matrices for plotting

        # cumulative differences
        sum_dz = sum_dz + dz
        sum_dmgs = sum_dmgs + dmgs
        sum_dsort = sum_dsort + dsort

        sum_dz_line[:,counter-1] = sum_dz
        sum_dmgs_line[:,counter-1] = sum_dmgs
        sum_dsort_line[:,counter-1] = sum_dsort

        # differences only; not cumulative
        dz_line[:,counter-1] = dz
        dmgs_line[:,counter-1] = dmgs
        dsort_line[:,counter-1] = dsort

        net_dz.append(np.nansum(dz)*3.)

        # undifferenced
        z_line[:,counter-1] = z
        mgs_line[:,counter-1] = mgs
        sort_line[:,counter-1] = sort

        # correlate dz with mgs(t-1), mgs, dmgs

        tmp_dz = dz
        tmp_dz = tmp_dz[~np.isnan(tmp_dz)] - np.nanmean(dz) # why this?

        tmp_mgs = mgs
        tmp_dmgs = dmgs
        tmp_sort = sort
        tmp_dsort = dsort
        tmp_last_mgs = last_mgs0
        tmp_last_sort = last_sort0

        tmp_last_dz = last_dz0
        # tmp_last_dz = tmp_last_dz[~np.isnan(tmp_last_dz)] - np.nanmean(last_dz0)

        plt_tag = 'mgs'
        tmp_mgs = tmp_mgs[~np.isnan(tmp_mgs)] - np.nanmean(mgs)

        # rr_dz_mgs, pp_dz_mgs, lo_dz_mgs, hi_dz_mgs = pearsonr_ci(all_dz,all_mgs,alpha=0.05)

        Ijnk = 0

        rtmp_dz_mgs, ptmp_dz_mgs, lotmp_dz_mgs, hitmp_dz_mgs = pearsonr_ci(tmp_dz[Ijnk:],tmp_mgs[Ijnk:len(tmp_dz)],alpha=0.05)
        rtmp_dz_dmgs, ptmp_dz_dmgs, lotmp_dz_dmgs, hitmp_dz_dmgs = pearsonr_ci(tmp_dz[Ijnk:],tmp_dmgs[Ijnk:len(tmp_dz)],alpha=0.05)
        rtmp_dz_sort, ptmp_dz_sort, lotmp_dz_sort, hitmp_dz_sort = pearsonr_ci(tmp_dz[Ijnk:],tmp_sort[Ijnk:len(tmp_dz)],alpha=0.05)
        rtmp_dz_dsort, ptmp_dz_dsort, lotmp_dz_dsort, hitmp_dz_dsort = pearsonr_ci(tmp_dz[Ijnk:],tmp_dsort[Ijnk:len(tmp_dz)],alpha=0.05)
        rtmp_dz_last_mgs, ptmp_dz_last_mgs, lotmp_dz_last_mgs, hitmp_dz_last_mgs = pearsonr_ci(tmp_dz[Ijnk:],tmp_last_mgs[Ijnk:len(tmp_dz)],alpha=0.05)
        rtmp_dz_last_sort, ptmp_dz_last_sort, lotmp_dz_last_sort, hitmp_dz_last_sort = pearsonr_ci(tmp_dz[Ijnk:],tmp_last_sort[Ijnk:len(tmp_dz)],alpha=0.05)
        rtmp_dz_last_dz, ptmp_dz_last_dz, lotmp_dz_last_dz, hitmp_dz_last_dz = pearsonr_ci(tmp_dz[Ijnk:],tmp_last_dz[Ijnk:len(tmp_dz)],alpha=0.05)

        corrcoeffs_dz_mgs.append(rtmp_dz_mgs)
        corrcoeffs_dz_dmgs.append(rtmp_dz_dmgs)
        corrcoeffs_dz_sort.append(rtmp_dz_sort)
        corrcoeffs_dz_dsort.append(rtmp_dz_dsort)
        corrcoeffs_dz_last_mgs.append(rtmp_dz_last_mgs)
        corrcoeffs_dz_last_sort.append(rtmp_dz_last_sort)
        corrcoeffs_dz_last_dz.append(rtmp_dz_last_dz)

        sig_dz_mgs.append(ptmp_dz_mgs)
        sig_dz_dmgs.append(ptmp_dz_dmgs)
        sig_dz_last_mgs.append(ptmp_dz_last_mgs)
        sig_dz_last_dz.append(ptmp_dz_last_dz)

        # corrcoeffs_dz_mgs.append(np.corrcoef(tmp_dz[Ijnk:],tmp_mgs[Ijnk:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_dmgs.append(np.corrcoef(tmp_dz[Ijnk:],tmp_dmgs[Ijnk:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_sort.append(np.corrcoef(tmp_dz[Ijnk:],tmp_sort[Ijnk:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_dsort.append(np.corrcoef(tmp_dz[Ijnk:],tmp_dsort[Ijnk:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_last_mgs.append(np.corrcoef(tmp_dz[Ijnk:],tmp_last_mgs[Ijnk:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_last_sort.append(np.corrcoef(tmp_dz[Ijnk:],tmp_last_sort[Ijnk:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_last_dz.append(np.corrcoef(tmp_dz[Ijnk:],tmp_last_dz[Ijnk:len(tmp_dz)])[0,1])



        # fig006, ax006 = plt.subplots(2,1, figsize=(9,9), num='spatial correlation (lower beach), tide-tide')
        # for mm in range(len(dz_line[1,:])):
        ax006[0,0].plot(tmp_dz, tmp_mgs[:len(tmp_dz)],'.')
        ax006[0,1].plot(tmp_dz, tmp_dmgs[:len(tmp_dz)],'.')
        ax006[1,0].plot(tmp_dz, tmp_sort[:len(tmp_dz)],'.')
        ax006[1,1].plot(tmp_dz ,tmp_dsort[:len(tmp_dz)],'.')
        ax006[2,0].plot(tmp_dz, tmp_last_mgs[:len(tmp_dz)],'.')
        ax006[2,1].plot(tmp_dz, tmp_last_sort[:len(tmp_dz)],'.')
        if counter > 1:
            ax006[2,2].plot(tmp_dz, tmp_last_dz[:len(tmp_dz)],'.')

        all_dz.extend(tmp_dz)
        all_mgs.extend(tmp_mgs[:len(tmp_dz)])
        all_dmgs.extend(tmp_dmgs[:len(tmp_dz)])
        all_last_mgs.extend(tmp_last_mgs[:len(tmp_dz)])

        if counter > 1:
            all_last_dz.extend(tmp_last_dz[:len(tmp_dz)])


        # Istart = 5
        # corrcoeffs_dz_mgs.append(np.corrcoef(tmp_dz[Istart:],tmp_mgs[Istart:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_dmgs.append(np.corrcoef(tmp_dz[Istart:],tmp_dmgs[Istart:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_sort.append(np.corrcoef(tmp_dz[Istart:],tmp_sort[Istart:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_dsort.append(np.corrcoef(tmp_dz[Istart:],tmp_dsort[Istart:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_last_mgs.append(np.corrcoef(tmp_dz[Istart:],tmp_last_mgs[Istart:len(tmp_dz)])[0,1])
        # corrcoeffs_dz_last_sort.append(np.corrcoef(tmp_dz[Istart:],tmp_last_sort[Istart:len(tmp_dz)])[0,1])

        ## cross-correlation
        # fig, ax1 = plt.subplots(2, 1, sharex=True, num='tide'+str(ii))
        # ax1[0].xcorr(tmp_dz, tmp_mgs[:len(tmp_dz)], usevlines=True, maxlags=10, normed=True, lw=2)
        # ax1[0].set_ylabel('correlation ($\Delta z, M_0$)')
        #
        # ax1[1].xcorr(tmp_dz, tmp_dmgs[:len(tmp_dz)], usevlines=True, maxlags=10, normed=True, lw=2)
        # ax1[1].set_ylabel('correlation ($\Delta z, \Delta M_0$)')
        # ax1[1].set_xlabel('lag')
        #
        # # EXPORT PLOTS
        # if saveCorr == 1:
        #     savedn = os.path.join(figsdn,'beach_profile',grid_spec,'cross_correlation',tide)
        #     savefn = 'dz_' + plt_tag
        #
        #     save_figures(savedn, savefn, fig)


    #tide 19: +1 more DGS obs than GPS
    if grid_spec == 'cross_shore':
        ax01[0].plot(y0, z0)
        ax01[1].plot(y0, dz[:len(y0)])
        ax01[2].plot(y0, mgs0[:len(y0)])
        ax01[3].plot(y0, sort0[:len(y0)])
    else:
        ax01[0].plot(x0, z0)
        ax01[1].plot(x0, dz[:len(y0)])
        ax01[2].plot(x0, mgs0[:len(y0)])
        ax01[3].plot(x0, sort0[:len(y0)])

    counter = counter + 1



# truncate nans
maxreal = 0
for col in sum_dz_line.T:
    candidate = np.count_nonzero(~np.isnan(col))
    if candidate > maxreal:
        maxreal = candidate

if grid_spec == "cross_shore":
    # apply loess regression to MGS, sorting
    smoothed_M0 = smooth_profile(mgs_line[:maxreal,:])
    smoothed_M1 = smooth_profile(sort_line[:maxreal,:])
    smoothed_dz = smooth_profile(sum_dz_line[:maxreal,:])
    smoothed_z = smooth_profile(z_line[:maxreal,:])



    # express dz as difference from linear fitted slope
    # z_lindiff = lin_fit_slope(z_line)


# FIGURES

ax01[0].set_ylabel('z [m]')
ax01[0].autoscale(enable=True, axis='x', tight=True)
ax01[1].set_ylabel(r'$\Delta z$ [mm]')
ax01[2].set_ylabel('mgs [mm]')
ax01[3].set_xlabel('cross-shore [m]')
ax01[3].set_ylabel('sorting [mm]')
fig01.tight_layout()


# plot change over time
fig1, ax1 = plt.subplots(3,1, figsize=(3,6), num='mgs, sort')
ax1_0 = ax1[0].imshow(sum_dz_line[:maxreal,:], cmap='bwr', vmin=-0.35, vmax=0.35, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb0 = fig1.colorbar(ax1_0, ax=ax1[0])
clb0.ax.set_title('$\Delta z$ [m]')
ax1_1 = ax1[1].imshow(mgs_line[:maxreal,:], cmap='inferno',extent=[tide_range[1],tide_range[-1],15,-30],aspect='auto')
clb1 = fig1.colorbar(ax1_1, ax=ax1[1])
clb1.ax.set_title('$M_0$ [mm]')
ax1_2 = ax1[2].imshow(sort_line[:maxreal,:], cmap='inferno',extent=[tide_range[1],tide_range[-1],15,-30],aspect='auto')
clb2 = fig1.colorbar(ax1_2, ax=ax1[2])
clb2.ax.set_title('$M_1$ [mm]')
fig1.tight_layout()


if grid_spec == "cross_shore":
    # plot change over time, smoothed, z lin fit
    fig001, ax001 = plt.subplots(3,1, figsize=(3.5,6.5), num='smoothed mgs, sort')
    # ax1_0 = ax001[0].imshow(sum_dz_line[:maxreal,:], cmap='bwr', vmin=-0.35, vmax=0.35, extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    ax1_0 = ax001[0].imshow(smoothed_dz, cmap='bwr', vmin=-0.35, vmax=0.35, extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    # ax1_0 = ax001[0].imshow(z_lindiff[:maxreal,:], cmap='bwr', vmin=-0.5, vmax=0.5, extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    # plot HWL locations
    ax001[0].plot(tide_axis,hwl,'kx')
    clb0 = fig001.colorbar(ax1_0, ax=ax001[0])
    clb0.ax.set_title('$\Delta z$ [m]', fontsize=10)
    ax001[0].set_xticklabels([])
    ax1_1 = ax001[1].imshow(smoothed_M0, cmap='inferno', vmin=5, vmax=35,extent=[tide_range[1],tide_range[-1],15,-30],aspect='auto')
    # plot HWL locations
    ax001[1].plot(tide_axis,hwl,'kx')
    clb1 = fig001.colorbar(ax1_1, ax=ax001[1])
    clb1.ax.set_title('$M_0$ [mm]', fontsize=10)
    ax001[1].set_xticklabels([])
    ax001[1].set_ylabel('distance offshore [m]')
    ax1_2 = ax001[2].imshow(smoothed_M1, cmap='inferno', vmin=5, vmax=35,extent=[tide_range[1],tide_range[-1],15,-30],aspect='auto')
    # plot HWL locations
    ax001[2].plot(tide_axis,hwl,'kx')
    clb2 = fig001.colorbar(ax1_2, ax=ax001[2])
    clb2.ax.set_title('$M_1$ [mm]', fontsize=10)
    ax001[2].set_xlabel('tide')
    # ax1_2.set_xlabel('cross-shore coordinate')
    fig001.tight_layout()


    fig010, ax010 = plt.subplots(1,1, figsize=(3.5,4.5), num='mean mgs, cross-shore')
    ax010.plot(np.mean(smoothed_M0, 1), np.linspace(-27, 9, 13), 'ko')
    ax010.plot([np.mean(smoothed_M0, 1)-np.std(smoothed_M0, 1), np.mean(smoothed_M0, 1)+np.std(smoothed_M0, 1)], [np.linspace(-27, 9, 13), np.linspace(-27, 9, 13)], 'k-')
    ax010.plot([np.min(smoothed_M0, 1), np.max(smoothed_M0, 1)], [np.linspace(-27, 9, 13), np.linspace(-27, 9, 13)], 'k|')
    ax010.axhspan(np.min(hwl), np.max(hwl), alpha=0.25, color='grey')
    xl = ax010.get_xlim()
    ax010.plot([xl[0],xl[1]], [-13,-13], 'k-.',  linewidth=1)
    ax010.plot([xl[0],xl[1]], [-5,-5], 'k--', linewidth=1)
    ax010.invert_yaxis()
    ax010.set_xlabel('MGS [mm]')
    ax010.set_ylabel('cross-shore coordinate, $y$ [m]')
    fig010.tight_layout()
    ax010.set_xlim(xl)


    # for surface plotting:
    Xtmp = np.linspace(tide_range[1], tide_range[-1], tide_range[-1]-tide_range[1]+1)
    Ytmp = np.linspace(-27, 9, 13)
    X, Y = np.meshgrid(Xtmp, Ytmp)
    Z = smoothed_z
    Zp = smoothed_dz

    jjj = matplotlib.colors.Normalize(vmin=-np.abs(np.max(smoothed_dz)), vmax=np.abs(np.max(smoothed_dz)))
    # jjj = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    vvv = jjj(smoothed_dz)
    cmap1=cm.bwr
    clrs1 = cmap1(vvv)

    hhh = matplotlib.colors.Normalize(vmin=0, vmax=np.abs(np.max(smoothed_M0)))
    uuu = hhh(smoothed_M0)
    cmap2=cm.inferno
    clrs2 = cmap2(uuu)

    # solve for vertical coordinate of HWL
    z_hwl = np.zeros(len(X[0,:]))
    for row in range(len(X[0,:])):
        hwl_now = hwl[row]
        Ihw = np.argmin(np.abs(hwl_now - Y[:,row]))
        z_hwl[row] = Z[Ihw, row]


    # surface plot ##########################################################
    fig = plt.figure(num='3d profiles', figsize=(5,6))

    acx1 = fig.add_subplot(2, 1, 1, projection='3d')
    surf = acx1.plot_surface(X, Y, Z, facecolors=clrs1, linewidth=0, antialiased=False)
    acx1.scatter(tide_axis, hwl,  z_hwl, marker='^', color='k', depthshade=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    # fig.colorbar(surf)
    # surf2 = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm,  linewidth=0.5, antialiased=False)
    acx1.invert_xaxis()
    acx1.view_init(elev=38., azim=50)
    # acx1.view_init(elev=38., azim=55)
    acx1.set_xlabel('low tide')
    acx1.set_ylabel('y [m]')
    acx1.set_zlabel('z [m]')

    acx2 = fig.add_subplot(2, 1, 2, projection='3d')
    surf = acx2.plot_surface(X, Y, Z, facecolors=clrs2, linewidth=0, antialiased=False)
    acx2.invert_xaxis()
    acx2.view_init(elev=38., azim=50)
    # acx2.view_init(elev=38., azim=55)
    acx2.scatter(tide_axis, hwl,  z_hwl, marker='^',color='k', depthshade=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    acx2.set_xlabel('low tide')
    acx2.set_ylabel('y [m]')
    acx2.set_zlabel('z [m]')

    a = smoothed_dz
    # plt.figure(figsize=(5, 5))
    fig.add_subplot(2, 1, 2)
    img = plt.imshow(a, cmap="bwr", vmin=-0.35, vmax=0.35)
    plt.gca().set_visible(False)
    cax = plt.axes([0.85, 0.595, 0.02, 0.25])
    clbjnk = plt.colorbar(cax=cax)
    # pl.savefig("colorbar.pdf")
    clbjnk.ax.set_title('$\Delta$z [m]', fontsize=10)

    a = smoothed_M0
    # plt.figure(figsize=(5, 5))
    fig.add_subplot(2, 1, 2)
    img = plt.imshow(a, cmap="inferno")
    plt.gca().set_visible(False)
    cax = plt.axes([0.85, 0.17, 0.02, 0.25])
    clbjnk = plt.colorbar(cax=cax)
    # pl.savefig("colorbar.pdf")
    clbjnk.ax.set_title('MGS [mm]', fontsize=10)

    ########################################################################


# clb2 = fig001.colorbar(ax1_2, ax=ax001[2])
# clb2.ax.set_title('$M_1$ [mm]', fontsize=10)

    # # temporal correlation of dz and DGS properties
    # fig002, ax002 = plt.subplots(2,1, figsize=(7,5), num='temporal correlation')
    # ax002[0].plot(tide_axis, smoothed_dz[-1,:])
    # ax002[1].plot(tide_axis, smoothed_M0[-1,:])


fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(3,6), num='cumulative change')
ax20 = ax2[0].imshow(sum_dz_line[:maxreal,:], cmap='bwr', vmin=-0.35, vmax=0.35, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb20 = fig2.colorbar(ax20, ax=ax2[0])
clb20.ax.set_title('$\Delta z$ [m]')
ax21 = ax2[1].imshow(sum_dmgs_line[:maxreal,:], cmap='bwr', vmin=-30, vmax=30, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb21 = fig2.colorbar(ax21, ax=ax2[1])
clb21.ax.set_title('$\Delta M_0$ [mm]')
ax22 = ax2[2].imshow(sum_dsort_line[:maxreal,:], cmap='bwr', vmin=-25, vmax=25, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb22 = fig2.colorbar(ax22, ax=ax2[2])
ax2[2].set_ylabel('cross-shore [m]')
ax2[2].set_xlabel('tide')
clb22.ax.set_title('$\Delta M_1$ [mm]')
fig2.tight_layout()

fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(3,6), num='tide-tide change change')
ax30 = ax3[0].imshow(dz_line[:maxreal,:], cmap='bwr', vmin=-0.2, vmax=0.2, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb30 = fig3.colorbar(ax30, ax=ax3[0])
clb30.ax.set_title('$\Delta z$ [m]')
ax31 = ax3[1].imshow(dmgs_line[:maxreal,:], cmap='bwr', vmin=-20, vmax=20, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb31 = fig3.colorbar(ax31, ax=ax3[1])
clb31.ax.set_title('$\Delta M_0$ [mm]')
ax32 = ax3[2].imshow(dsort_line[:maxreal,:], cmap='bwr', vmin=-15, vmax=15, \
extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
clb32 = fig3.colorbar(ax32, ax=ax3[2])
ax3[2].set_ylabel('cross-shore [m]')
ax3[2].set_xlabel('tide')
clb32.ax.set_title('$\Delta M_1$ [mm]')
fig3.tight_layout()

# np.corrcoef(tmp_dz,tmp_mgs[:len(tmp_dz)])[0,1]

# temporal_corrcoeffs_dz_dmgs = []
# temporal_corrcoeffs_dz_mgs = []
# for nn in range(len(dz_line[:,1])):
#     temporal_corrcoeffs_dz_dmgs.append(np.corrcoef(dz_line[nn,:],dmgs_line[nn,:])[0,1])
#     temporal_corrcoeffs_dz_mgs.append(np.corrcoef(dz_line[nn,:],mgs_line[nn,:])[0,1])
#     # temporal_dz_dmgs.extend(dz_line[nn,:],dmgs_line[nn,:])
#     # temporal_dz_mgs.extend(dz_line[nn,:],mgs_line[nn,:])
# # temporal correlation of dz and DGS properties
# fig003, ax003 = plt.subplots(4,1, figsize=(7,9), num='temporal correlation, tide-tide')
# ax003[0].plot(temporal_corrcoeffs)
# ax003[1].plot(tide_axis, dz_line[maxreal,:])
# ax003[2].plot(tide_axis, dmgs_line[maxreal,:])
# ax003[3].plot(dz_line[:maxreal,:], dmgs_line[:maxreal,:],'.')
# # np.nanmean(temporal_corrcoeffs_dz_dmgs)
# # np.nanstd(temporal_corrcoeffs_dz_dmgs)
# # np.nanmean(temporal_corrcoeffs_dz_mgs)
# # np.nanstd(temporal_corrcoeffs_dz_mgs)

# temporal_corrcoeffs_dz_mgs = ~np.isnan(np.array(temporal_corrcoeffs_dz_mgs))

# fig007, ax007 = plt.subplots(1,1, figsize=(7,4), num='temporal correlation, by spatial coordinate')
# ax007.plot(temporal_corrcoeffs_dz_mgs[:maxreal],np.arange(15, 28), '.')
# ax007.plot(temporal_corrcoeffs_dz_dmgs[:maxreal],np.arange(15, 28), '.')
# ax007.invert_yaxis()

fig004, ax004 = plt.subplots(2,1, figsize=(9,9), num='temporal correlation (lower beach), tide-tide')
ax004[0].plot(dz_line[:,:], dmgs_line[:,:],'.')
ax004[1].plot(dz_line[:,:], mgs_line[:,:],'.')

pearsonr_dz_mgs, alpha_dz_mgs = pearsonr(all_dz, all_mgs)
pearsonr_dz_dmgs, alpha_dz_dmgs = pearsonr(all_dz, all_dmgs)
pearsonr_dz_last_mgs, alpha_dz_last_mgs = pearsonr(all_dz, all_last_mgs)
pearsonr_dz_last_dz, alpha_dz_last_dz = pearsonr(all_dz[len(all_dz) - len(all_last_dz):], all_last_dz)

rr_dz_mgs, pp_dz_mgs, lo_dz_mgs, hi_dz_mgs = pearsonr_ci(all_dz,all_mgs,alpha=0.05)
rr_dz_dmgs, pp_dz_dmgs, lo_dz_dmgs, hi_dz_dmgs = pearsonr_ci(all_dz,all_dmgs,alpha=0.05)
rr_dz_last_mgs, pp_dz_last_mgs, lo_dz_last_mgs, hi_dz_last_mgs = pearsonr_ci(all_dz,all_last_mgs,alpha=0.05)
rr_dz_last_dz, pp_dz_last_dz, lo_dz_last_dz, hi_dz_last_dz = pearsonr_ci(all_dz[len(all_dz) - len(all_last_dz):],all_last_dz,alpha=0.05)


# fig005, ax005 = plt.subplots(2,1, figsize=(9,9), num='spatial correlation (lower beach), tide-tide')
# # for mm in range(len(dz_line[1,:])):
# ax005[0].plot(tmp_dz, tmp_mgs[:len(tmp_dz)],'.')
# ax005[1].plot(dz_line[:,1], mgs_line[:,1],'.')


# correlation vs hydrodynamics

# significant wave height
# fit data
fit1, r1, tmpx = linear_regression(corrcoeffs_dz_mgs, Hs[1:])
fit2, r2, tmpx = linear_regression(corrcoeffs_dz_dmgs, Hs[1:])
# plot
fig4 = plt.figure(num='Hs')
plt.plot(Hs[1:], corrcoeffs_dz_mgs, 'k.')
plt.plot(Hs[1:], corrcoeffs_dz_dmgs, 'r.')
plt.plot(tmpx, np.polyval(fit1, tmpx), 'k-')
plt.plot(tmpx, np.polyval(fit2, tmpx), 'r-')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Hs [m]')
plt.ylabel('correlation coeff.')
plt.legend([r'$\Delta z,M_0, r^2=$'+str(r1), r'$\Delta z,\Delta M_0, r^2=$'+str(r2)])

# wind band energy
# fit data
fit1, r1, tmpx = linear_regression(corrcoeffs_dz_mgs, wave_energy_wind[1:])
fit2, r2, tmpx = linear_regression(corrcoeffs_dz_dmgs, wave_energy_wind[1:])
# plot
fig5 = plt.figure(num='energy')
plt.plot(wave_energy_wind[1:], corrcoeffs_dz_mgs, 'k.')
plt.plot(wave_energy_wind[1:], corrcoeffs_dz_dmgs, 'r.')
plt.plot(tmpx, np.polyval(fit1, tmpx), 'k-')
plt.plot(tmpx, np.polyval(fit2, tmpx), 'r-')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('wave energy - wind band [m^2]')
plt.ylabel('correlation coeff.')
plt.legend([r'$\Delta z,M_0, r^2=$'+str(r1), r'$\Delta z,\Delta M_0, r^2=$'+str(r2)])

# peak period
# fit data
fit1, r1, tmpx = linear_regression(corrcoeffs_dz_mgs, Tp[1:])
fit2, r2, tmpx = linear_regression(corrcoeffs_dz_dmgs, Tp[1:])
# plot
fig6 = plt.figure(num='Tp')
plt.plot(Tp[1:], corrcoeffs_dz_mgs, 'k.')
plt.plot(Tp[1:], corrcoeffs_dz_dmgs, 'r.')
plt.plot(tmpx, np.polyval(fit1, tmpx), 'k-')
plt.plot(tmpx, np.polyval(fit2, tmpx), 'r-')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('$T_p$ [s]')
plt.ylabel('correlation coeff.')
plt.legend([r'$\Delta z,M_0, r^2=$'+str(r1), r'$\Delta z,\Delta M_0, r^2=$'+str(r2)])

# peak period
# fit data
fit1, r1, tmpx = linear_regression(corrcoeffs_dz_mgs, steepness[1:])
fit2, r2, tmpx = linear_regression(corrcoeffs_dz_dmgs, steepness[1:])
# plot
fig7 = plt.figure(num='steep')
plt.plot(steepness[1:], corrcoeffs_dz_mgs, 'k.')
plt.plot(steepness[1:], corrcoeffs_dz_dmgs, 'r.')
plt.plot(tmpx, np.polyval(fit1, tmpx), 'k-')
plt.plot(tmpx, np.polyval(fit2, tmpx), 'r-')
plt.xlabel('steepness')
plt.ylabel('correlation coeff.')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend([r'$\Delta z,M_0, r^2=$'+str(r1), r'$\Delta z,\Delta M_0, r^2=$'+str(r2)])

# peak period
# fit data
fit1, r1, tmpx = linear_regression(corrcoeffs_dz_mgs, iribarren[1:])
fit2, r2, tmpx = linear_regression(corrcoeffs_dz_dmgs, iribarren[1:])
# plot
fig8 = plt.figure(num='iribarren')
plt.plot(iribarren[1:], corrcoeffs_dz_mgs, 'k.')
plt.plot(iribarren[1:], corrcoeffs_dz_dmgs, 'r.')
plt.plot(tmpx, np.polyval(fit1, tmpx), 'k-')
plt.plot(tmpx, np.polyval(fit2, tmpx), 'r-')
plt.xlabel('Iribarren')
plt.ylabel('correlation coeff.')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend([r'$\Delta z,M_0, r^2=$'+str(r1), r'$\Delta z,\Delta M_0, r^2=$'+str(r2)])


mean_cc_dz_mgs = np.mean(corrcoeffs_dz_mgs)
std_cc_dz_mgs = np.std(corrcoeffs_dz_mgs)
mean_cc_dz_dmgs = np.mean(corrcoeffs_dz_dmgs)
std_cc_dz_dmgs = np.std(corrcoeffs_dz_dmgs)
mean_cc_dz_sort = np.mean(corrcoeffs_dz_sort)
std_cc_dz_sort = np.std(corrcoeffs_dz_sort)
mean_cc_dz_dsort = np.mean(corrcoeffs_dz_dsort)
std_cc_dz_dsort = np.std(corrcoeffs_dz_dsort)
mean_cc_dz_last_mgs = np.mean(corrcoeffs_dz_last_mgs)
std_cc_dz_last_mgs = np.std(corrcoeffs_dz_last_mgs)
mean_cc_dz_last_sort = np.mean(corrcoeffs_dz_last_sort)
std_cc_dz_last_sort = np.std(corrcoeffs_dz_last_sort)
mean_cc_dz_last_dz = np.nanmean(corrcoeffs_dz_last_dz)
std_cc_dz_last_dz = np.nanstd(corrcoeffs_dz_last_dz)

fig9, ax9 = plt.subplots(2,1,figsize=(3,7), gridspec_kw={'height_ratios': [3, 1]}, num='correlation coefficients')
ax9[0].plot([0,0],[13,28],'k--')
ax9[0].plot(corrcoeffs_dz_mgs, tide_axis, 'C0.')
ax9[0].plot(corrcoeffs_dz_sort, tide_axis, 'C1.')
ax9[0].plot(corrcoeffs_dz_dmgs, tide_axis, 'C2.')
ax9[0].plot(corrcoeffs_dz_dsort, tide_axis, 'C3.')
ax9[0].plot(corrcoeffs_dz_last_mgs, tide_axis, 'C4.')
ax9[0].plot(corrcoeffs_dz_last_sort, tide_axis, 'C5.')
ax9[0].plot(corrcoeffs_dz_last_dz, tide_axis, 'C6.')
ax9[0].autoscale(enable=True, axis='y', tight=True)
ax9[0].invert_yaxis()
ax9[0].set_ylabel('tide')
ax90xlim = ax9[0].get_xlim()

ax9[1].plot([0,0],[-0.5,6.5],'k--')
ax9[1].plot(mean_cc_dz_mgs, 0, 'C0.')
ax9[1].plot([mean_cc_dz_mgs-std_cc_dz_mgs,mean_cc_dz_mgs+std_cc_dz_mgs], [0,0], 'C0-')
ax9[1].plot(mean_cc_dz_sort, 1, 'C1.')
ax9[1].plot([mean_cc_dz_sort-std_cc_dz_sort,mean_cc_dz_sort+std_cc_dz_sort], [1,1], 'C1-')
ax9[1].plot(mean_cc_dz_dmgs, 2, 'C2.')
ax9[1].plot([mean_cc_dz_dmgs-std_cc_dz_dmgs,mean_cc_dz_dmgs+std_cc_dz_dmgs], [2,2], 'C2-')
ax9[1].plot(mean_cc_dz_dsort, 3, 'C3.')
ax9[1].plot([mean_cc_dz_dsort-std_cc_dz_dsort,mean_cc_dz_dsort+std_cc_dz_dsort], [3,3], 'C3-')
ax9[1].plot(mean_cc_dz_last_mgs, 4, 'C4.')
ax9[1].plot([mean_cc_dz_last_mgs-std_cc_dz_last_mgs,mean_cc_dz_last_mgs+std_cc_dz_last_mgs], [4,4], 'C4-')
ax9[1].plot(mean_cc_dz_last_sort, 5, 'C5.')
ax9[1].plot([mean_cc_dz_last_sort-std_cc_dz_last_sort,mean_cc_dz_last_sort+std_cc_dz_last_sort], [5,5], 'C5-')
ax9[1].plot(mean_cc_dz_last_dz, 6, 'C6.')
ax9[1].plot([mean_cc_dz_last_dz-std_cc_dz_last_dz,mean_cc_dz_last_dz+std_cc_dz_last_dz], [6,6], 'C6-')
ax9[1].autoscale(enable=True, axis='y', tight=True)
ax9[1].invert_yaxis()
ax9[1].set_xlabel('correlation coefficient')
setstr = [r'$\Delta z,M_0$', r'$\Delta z,M_1$', r'$\Delta z,\Delta M_0$',\
r'$\Delta z,\Delta M_1$', r'$\Delta z,M_0[t-1]$', r'$\Delta z,M_1[t-1]$', r'$\Delta z,\Delta z[t-1]$']
setind = [0, 1, 2, 3, 4, 5, 6]
ax9[1].set_yticks(setind)
ax9[1].set_yticklabels(setstr)
ax9[1].set_xlim(ax90xlim)
fig9.tight_layout()

# plot changes in profile (z, mgs, ...) against hydrodynamics
fig10, ax10 = plt.subplots(3,1,figsize=(4,9), num='profile change')
ax10[0].plot(Hs[1:], np.nanmean(dz_line,axis=0), '.')
ax10[0].set_ylabel(r'$\Delta z$ [m]')
ax10[1].plot(Hs[1:], np.nanmean(mgs_line,axis=0), '.')
ax10[1].set_ylabel(r'$M_0$ [mm]')
ax10[2].plot(Hs[1:], np.nanmean(sort_line,axis=0), '.')
ax10[2].set_ylabel(r'$M_1$ [mm]')
ax10[2].set_xlabel('H_s [m]')
fig10.tight_layout()

# plot changes in profile (z, mgs, ...) against hydrodynamics
fig11, ax11 = plt.subplots(2,1,figsize=(5,3), sharex=True,num='Hs and grainsize change')
ax11[0].plot(tide_axis, Hs[1:], 'k.')
ax11[0].set_ylabel('$H_s$ [m]')
ax11[0].tick_params(direction='in',top=1,right=1)
ax11[1].plot(tide_axis, np.nanmean(mgs_line,axis=0), 'k.')
ax11[1].plot([tide_axis, tide_axis], [np.nanmean(mgs_line,axis=0) - np.nanstd(mgs_line,axis=0), np.nanmean(mgs_line,axis=0) + np.nanstd(mgs_line,axis=0)], 'k-')
# ax11[1].errorbar(tide_axis, np.nanmean(mgs_line,axis=0),
#             xerr=0,
#             yerr=np.nanmean(sort_line,axis=0))
ax11[1].set_ylabel('$M_0$ [mm]')
ax11[1].set_xlabel('tide')
ax11[1].tick_params(direction='in',top=1,right=1)
fig11.tight_layout()


# plot correlation coefficients against high tide elevation
# plot changes in profile (z, mgs, ...) against hydrodynamics
fig12, ax12 = plt.subplots(3,1,figsize=(5,9), num='correlation against high tide elevation')
# ax12[0].plot(maxdepth[1:], corrcoeffs_dz_mgs, '.')
ax12[0].plot(hwl, corrcoeffs_dz_mgs, '.')
ax12[0].set_ylabel(r'$R^2 (\Delta z,M_0)$')
# ax12[1].plot(maxdepth[1:], corrcoeffs_dz_dmgs, '.')
ax12[1].plot(hwl, corrcoeffs_dz_dmgs, '.')
ax12[1].set_ylabel(r'$R^2 (\Delta z,\Delta M_0)$')
ax12[1].set_xlabel('max depth [m]')
fig12.tight_layout()

plt.figure(453)
plt.plot(maxdepth, 'o')

fig987, ax987 = plt.subplots(2,1,num='steepness vs vol change')
ax987[0].plot(tide_axis, net_dz, '.')
ax987[0].set_ylabel('vol change [m^2]')
ax987[1].plot(tide_axis, steepness[1:], '.')
ax987[1].set_ylabel('H/L')
ax987[1].set_xlabel('tide')

# EXPORT PLOTS
if saveFlag == 1:

    # savedn = os.path.join(figsdn,'beach_profile',grid_spec)
    savedn = os.path.join(figsdn,'beach_profile','reprocessed',grid_spec)

    save_figures(savedn, 'surfaceplots_dz_grainsize', fig)

    save_figures(savedn, 'mean_MGS_crossshore', fig010)

    # save_figures(savedn, 'elevation_and_grainsize_smoothed', fig001)
    #
    # save_figures(savedn, 'elevation_and_grainsize', fig01)
    # save_figures(savedn, 'cumulative_elevation_and_grainsize', fig1)
    # save_figures(savedn, 'cumulative_elevation_and_grainsize_change', fig2)
    # save_figures(savedn, 'tidal_elevation_and_grainsize_change', fig3)
    # save_figures(savedn, 'Hs_corr_coeff', fig4)
    # save_figures(savedn, 'energy_corr_coeff', fig5)
    # save_figures(savedn, 'Tp_corr_coeff', fig6)
    # save_figures(savedn, 'steepness_corr_coeff', fig7)
    # save_figures(savedn, 'iribarren_corr_coeff', fig8)
    save_figures(savedn, 'pearson_correlation_coefficients', fig9)
    # save_figures(savedn, 'profile_change', fig10)
    # save_figures(savedn, 'grain_size_and_waveheight_timeseries', fig11)


# if __name__ == '__main__':
#     main()
