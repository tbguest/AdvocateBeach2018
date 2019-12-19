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

    corr_data = r, p, lo, hi

    # return r, p, lo, hi
    return corr_data

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



# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux

# dn_in = os.path.join(homechar,'Projects','AdvocateBeach2018','data', 'processed', 'survey_data')
dn_in = os.path.join(homechar,'Projects','AdvocateBeach2018','data', 'processed', 'survey_data', 'reprocessed_x10')

grid_specs = ["cross_shore","longshore1", "longshore2", "dense_array2"]
# grid_specs = ["longshore2","longshore1"]
# grid_specs = ["dense_array2"]#,"longshore2"]
# grid_specs = ["dense_array2"]

# 0 through  5
dense_array_row = 5



fig1, ax1 = plt.subplots(1,2,figsize=(6,3), num='tide only correlations')
fig01, ax01 = plt.subplots(1,1,figsize=(3,2.5), num='tide only correlations - for RCEM')
fig2, ax2 = plt.subplots(1,2,figsize=(6,3), num='scatter')

foobar = []

for grid_spec in grid_specs:
    # grid_spec = "longshore1"
    # grid_spec = "longshore2"
    # grid_spec = "cross_shore"
    # grid_spec = "dense_array2"

    # if grid_spec is dense_array2:
    #     isolate rows


    fn_in = os.path.join(dn_in, grid_spec + '.npy')

    jnk = np.load(fn_in, allow_pickle=True).item()

    # jnk['dz']

    dz_all = jnk['dz']
    mgs_all = jnk['mgs']
    dmgs_all = jnk['dmgs']
    y = jnk['y']

    # NEED TO SAVE XY coords also
    # then, can introdoce conditions involving HWL

    if grid_spec == 'cross_shore':
        start_tide = 14
    elif grid_spec == 'longshore1':
        start_tide = 16
    elif grid_spec == 'dense_array2':
        start_tide = 19
    else:
        start_tide = 15

    tide_range = range(start_tide, 28)
    tide_axis = np.arange(start_tide+1,28) # for plotting later

    # wave data
    Hs_array = []
    Tp_array = []
    steepness_array = []
    iribarren_array = []
    wave_energy_array = []
    maxdepth_array = []
    # hwl = []

    corr1 = []
    corr2 = []
    corr3 = []
    corr4 = []
    corr5 = []
    corr6 = []
    corr4b = []

    mean_dz_tide = []
    mean_mgs_tide = []
    mean_dmgs_tide = []

    dz_mean_removed = []
    dmgs_mean_removed = []
    dz_mean_removed_lowenergy = []
    dz_mean_removed_highenergy = []
    dmgs_mean_removed_lowenergy = []
    dmgs_mean_removed_highenergy = []

    counter = -1

    for ii in tide_range:

        counter += 1

        if not ii in dz_all:
            continue

        hwl = jnk['hwl'][ii]

        if grid_spec == 'dense_array2':

            dz_tide = np.array(dz_all[ii]).reshape(6,24)[dense_array_row,:]
            mgs_tide = np.array(mgs_all[ii]).reshape(6,24)[dense_array_row,:]
            dmgs_tide = np.array(dmgs_all[ii]).reshape(6,24)[dense_array_row,:]
            y1 = np.array(y).reshape(6,24)[dense_array_row,:]

            if counter > 0:
                last_dz_tide = np.array(dz_all[ii-1]).reshape(6,24)[dense_array_row,:]
                last_mgs_tide = np.array(mgs_all[ii-1]).reshape(6,24)[dense_array_row,:]
                last_dmgs_tide = np.array(dmgs_all[ii-1]).reshape(6,24)[dense_array_row,:]

        else:


            dz_tide = dz_all[ii]
            mgs_tide = mgs_all[ii]
            dmgs_tide = dmgs_all[ii]
            y1 = np.array(y)

            if counter > 0:
                last_dz_tide = dz_all[ii-1]
                last_mgs_tide = mgs_all[ii-1]
                last_dmgs_tide = dmgs_all[ii-1]




        # For end of script correlation
        mean_dz_tide.append(np.mean(dz_tide))
        mean_mgs_tide.append(np.mean(mgs_tide))
        mean_dmgs_tide.append(np.mean(dmgs_tide))


        if y1[0] > jnk['hwl'][ii]: # omit stations above hwl
            foobar.append(ii)
            dz_mean_removed.extend(np.array(dz_tide) - np.mean(dz_tide))
            dmgs_mean_removed.extend(np.array(dmgs_tide) - np.mean(dmgs_tide))
            if counter > 0:
                dz_mean_removed0 = np.array(dz_tide) - np.mean(dz_tide)
                last_dz_mean_removed0 = np.array(last_dz_tide) - np.mean(last_dz_tide)


        if jnk['steepness'][ii] < 0.01:
            dz_mean_removed_lowenergy.extend(np.array(dz_tide) - np.mean(dz_tide))
            dmgs_mean_removed_lowenergy.extend(np.array(dmgs_tide) - np.mean(dmgs_tide))
        else:
            dz_mean_removed_highenergy.extend(np.array(dz_tide) - np.mean(dz_tide))
            dmgs_mean_removed_highenergy.extend(np.array(dmgs_tide) - np.mean(dmgs_tide))

        # this needs work if i wnt it to index properly
        Hs_array.append(jnk['Hs'][ii])
        Tp_array.append(jnk['Tp'][ii])
        steepness_array.append(jnk['steepness'][ii])
        iribarren_array.append(jnk['iribarren'][ii])
        wave_energy_array.append(jnk['wave_energy'][ii])
        maxdepth_array.append(jnk['maxdepth'][ii])


        # tide-only correlations
        # r1, p1, lo1, hi1 = pearsonr_ci(dz_tide, mgs_tide, alpha=0.05)
        # r2, p2, lo2, hi2 = pearsonr_ci(dz_tide, dmgs_tide, alpha=0.05)
        # r3, p3, lo3, hi3 = pearsonr_ci(dz_tide[1:], mgs_tide[:-1], alpha=0.05)
        # r4, p4, lo4, hi4 = pearsonr_ci(dz_tide[1:], dz_tide[:-1], alpha=0.05)
        # r5, p5, lo5, hi5 = pearsonr_ci(mgs_tide[1:], mgs_tide[:-1], alpha=0.05)
        # r6, p6, lo6, hi6 = pearsonr_ci(dmgs_tide[1:], dmgs_tide[:-1], alpha=0.05)
        r1 = pearsonr_ci(dz_tide, mgs_tide, alpha=0.05)
        r2 = pearsonr_ci(dz_tide, dmgs_tide, alpha=0.05)

# plt.figure()
# plt.plot(dz_tide)
# plt.plot(dz_mean_removed0)
#


        if counter > 0:
            r3 = pearsonr_ci(dz_tide, last_mgs_tide, alpha=0.05)
            # r4 = pearsonr_ci(dz_tide, last_dz_tide, alpha=0.05)
            if y1[0] > jnk['hwl'][ii]:
                r4 = pearsonr_ci(dz_mean_removed0, last_dz_mean_removed0, alpha=0.05)
            r5 = pearsonr_ci(mgs_tide, last_mgs_tide, alpha=0.05)
            r6 = pearsonr_ci(dmgs_tide, last_dmgs_tide, alpha=0.05)
            # recompute r4 with |dz|<0.02
            Ir4_1 = np.asarray(np.where(np.abs(np.array(dz_tide)) > 0.02)).squeeze().tolist()
            Ir4_2 = np.asarray(np.where(np.abs(np.array(last_dz_tide)) > 0.02)).squeeze().tolist()
            Icommon = list(set(Ir4_1).intersection(Ir4_2))
            r4b = pearsonr_ci(np.array(dz_tide)[Icommon], np.array(last_dz_tide)[Icommon], alpha=0.05)

            # append to list to disentangle and plot later
            corr3.append(r3)
            if y1[0] > jnk['hwl'][ii]:
                corr4.append(r4)
            corr5.append(r5)
            corr6.append(r6)
            if len(Icommon) > 4:
                corr4b.append(r4b)

        # append to list to disentangle and plot later
        corr1.append(r1)
        corr2.append(r2)


    # tides only - spatial
    r1_tides = [corr1[n][0] for n in range(len(corr1))]
    r2_tides = [corr2[n][0] for n in range(len(corr2))]
    r3_tides = [corr3[n][0] for n in range(len(corr3))]
    if y1[0] > jnk['hwl'][ii]:
        r4_tides = [corr4[n][0] for n in range(len(corr4))]
    r5_tides = [corr5[n][0] for n in range(len(corr5))]
    r6_tides = [corr6[n][0] for n in range(len(corr6))]
    r4b_tides = [corr4b[n][0] for n in range(len(corr4b))]

    # compiled dz, dmgs data; means removed
    r1_all = pearsonr_ci(dz_mean_removed, dmgs_mean_removed, alpha=0.05)

    # wave energy-dependent
    r1_all_lowenergy = pearsonr_ci(dz_mean_removed_lowenergy, dmgs_mean_removed_lowenergy, alpha=0.05)
    r1_all_highenergy = pearsonr_ci(dz_mean_removed_highenergy, dmgs_mean_removed_highenergy, alpha=0.05)

    # time only; space averaged
    r1_time = pearsonr_ci(mean_dz_tide, mean_mgs_tide, alpha=0.05)
    r2_time = pearsonr_ci(mean_dz_tide, mean_dmgs_tide, alpha=0.05)
    rwaves11 = pearsonr_ci(mean_dz_tide, Hs_array, alpha=0.05)
    rwaves12 = pearsonr_ci(mean_dz_tide, Tp_array, alpha=0.05)
    rwaves13 = pearsonr_ci(mean_dz_tide, steepness_array, alpha=0.05)
    rwaves14 = pearsonr_ci(mean_dz_tide, iribarren_array, alpha=0.05)
    rwaves21 = pearsonr_ci(mean_mgs_tide, Hs_array, alpha=0.05)
    rwaves22 = pearsonr_ci(mean_mgs_tide, Tp_array, alpha=0.05)
    rwaves23 = pearsonr_ci(mean_mgs_tide, steepness_array, alpha=0.05)
    rwaves24 = pearsonr_ci(mean_mgs_tide, iribarren_array, alpha=0.05)
    rwaves31 = pearsonr_ci(mean_dmgs_tide, Hs_array, alpha=0.05)
    rwaves32 = pearsonr_ci(mean_dmgs_tide, Tp_array, alpha=0.05)
    rwaves33 = pearsonr_ci(mean_dmgs_tide, steepness_array, alpha=0.05)
    rwaves34 = pearsonr_ci(mean_dmgs_tide, iribarren_array, alpha=0.05)
# np.mean(Hs_array)
# np.mean(Tp_array)
    outFlag = 1
    if outFlag == 1:

        output = {'correlation':r1_all, 'y':y1[0], 'correlation_lowenergy':r1_all_lowenergy, 'correlation_highenergy':r1_all_highenergy}
        # save new variables
        # fout = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
        #                  "processed","survey_data", "dz_dmgs_correlations")
        fout = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                         "processed","survey_data", "reprocessed_x10", "dz_dmgs_correlations")

        if not os.path.exists(fout):
            try:
                os.makedirs(fout)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        if grid_spec == 'dense_array2':
            fn = grid_spec + '-' + str(dense_array_row)
        else:
            fn = grid_spec
        np.save(os.path.join(fout, fn), output)



    fig0, ax0 = plt.subplots(3,2,figsize=(6,7.5), num='wave + MSD scatter plots')
    ax0[0,0].plot(mean_dz_tide, mean_mgs_tide, 'k.')
    ax0[0,1].plot(mean_dz_tide, mean_dmgs_tide, 'k.')
    ax0[1,0].plot(mean_dz_tide, Hs_array, 'k.')
    ax0[1,1].plot(mean_mgs_tide, Hs_array, 'k.')
    ax0[2,0].plot(mean_dz_tide, steepness_array, 'k.')
    ax0[2,1].plot(mean_mgs_tide, steepness_array, 'k.')
    fig0.tight_layout()



    if grid_spec == 'longshore2':
        ax1[0].plot([0,0],[-0.5,3.5],'k--')
        ax1[0].plot(r1_tides, 0*np.ones(len(r1_tides)), 'kx')
        ax1[0].plot(r2_tides, 1*np.ones(len(r2_tides)), 'kx')
        ax1[0].plot(r3_tides, 2*np.ones(len(r3_tides)), 'kx')
        ax1[0].plot(r4_tides, 3*np.ones(len(r4_tides)), 'kx')
        ax1[0].plot(r4b_tides, 3*np.ones(len(r4b_tides)), 'rx')
        # ax1.plot(r5_tides, 4*np.ones(len(r5_tides)), 'kx')
        # ax1.plot(r6_tides, 5*np.ones(len(r6_tides)), 'kx')
        ax1[0].autoscale(enable=True, axis='y', tight=True)
        ax1[0].invert_yaxis()
        ax1[0].set_xlabel('correlation coefficient')
        setstr = [r'$\Delta z$,MGS', r'$\Delta z$,$\Delta$MGS', r'$\Delta z$,MGS$_{t-1}$', \
                r'$\Delta z$,$\Delta z_{t-1}$']#, r'$mgs,mgs_{t-1}$', r'$\Delta mgs,\Delta mgs_{t-1}$']
        setind = [0, 1, 2, 3]#, 4, 5]
        ax1[0].set_yticks(setind)
        ax1[0].set_yticklabels(setstr)
        # ax9[1].set_xlim(ax90xlim)
    elif grid_spec == 'longshore1':
        ax1[1].plot([0,0],[-0.5,3.5],'k--')
        ax1[1].plot(r1_tides, 0*np.ones(len(r1_tides)), 'kx')
        ax1[1].plot(r2_tides, 1*np.ones(len(r2_tides)), 'kx')
        ax1[1].plot(r3_tides, 2*np.ones(len(r3_tides)), 'kx')
        ax1[1].plot(r4_tides, 3*np.ones(len(r4_tides)), 'kx')
        ax1[1].plot(r4b_tides, 3*np.ones(len(r4b_tides)), 'rx')
        # ax1.plot(r5_tides, 4*np.ones(len(r5_tides)), 'kx')
        # ax1.plot(r6_tides, 5*np.ones(len(r6_tides)), 'kx')
        ax1[1].autoscale(enable=True, axis='y', tight=True)
        ax1[1].invert_yaxis()
        ax1[1].set_xlabel('correlation coefficient')
        setstr = [r'$\Delta z$,MGS', r'$\Delta z$,$\Delta$MGS', r'$\Delta z$,MGS$_{t-1}$', \
                r'$\Delta z$,$\Delta z_{t-1}$']#, r'$mgs,mgs_{t-1}$', r'$\Delta mgs,\Delta mgs_{t-1}$']
        setind = [0, 1, 2, 3]#, 4, 5]
        ax1[1].set_yticks(setind)
        ax1[1].set_yticklabels([])
        # ax1[1].set_yticklabels(setstr)
        # ax9[1].set_xlim(ax90xlim)
    fig1.tight_layout()


    if grid_spec == 'longshore1':
        ax01.plot([0,0],[-0.5,2.5],'k--')
        # ax01[1].plot(r1_tides, 0*np.ones(len(r1_tides)), 'kx')
        ax01.plot(r2_tides, 0*np.ones(len(r2_tides)), 'kx')
        ax01.plot(r3_tides, 1*np.ones(len(r3_tides)), 'kx')
        ax01.plot(r4_tides, 2*np.ones(len(r4_tides)), 'kx')
        # ax01[1].plot(r4b_tides, 3*np.ones(len(r4b_tides)), 'rx')
        # ax1.plot(r5_tides, 4*np.ones(len(r5_tides)), 'kx')
        # ax1.plot(r6_tides, 5*np.ones(len(r6_tides)), 'kx')
        ax01.autoscale(enable=True, axis='y', tight=True)
        ax01.invert_yaxis()
        ax01.set_xlabel('correlation coefficient')
        setstr = [r'$\Delta z$,$\Delta$MGS', r'$\Delta z$,MGS$_{t-1}$', \
                r'$\Delta z$,$\Delta z_{t-1}$']#, r'$mgs,mgs_{t-1}$', r'$\Delta mgs,\Delta mgs_{t-1}$']
        setind = [0, 1, 2]#, 4, 5]
        ax01.set_yticks(setind)
        ax01.set_yticklabels([])
        # ax1[1].set_yticklabels(setstr)
        # ax9[1].set_xlim(ax90xlim)
        fig01.tight_layout()


    # fig2, ax2 = plt.subplots(1,2,figsize=(6,4), num='scatter')
    if grid_spec == 'longshore2':
        ax2[0].plot(dz_mean_removed, dmgs_mean_removed, 'C0.')
        ax2[0].grid()
        ax2[0].set_ylabel(r'$\Delta$MGS [mm]')
        ax2[0].set_xlabel(r'$\Delta z$ [m]')
        # yl = ax2[0].get_ylim()
        xl = ax2[0].get_xlim()
    elif grid_spec == 'longshore1':
        ax2[1].plot(dz_mean_removed, dmgs_mean_removed, 'C1.')
        ax2[1].grid()
        # ax2[1].set_ylabel(r'$\Delta$MGS [mm]')
        ax2[1].set_xlabel(r'$\Delta z$ [m]')
        yl = ax2[1].get_ylim()
        ax2[0].set_ylim(yl)
        # xl = ax2[0].get_xlim()
        ax2[1].set_xlim(xl)
    fig2.tight_layout()


    #
    #
    # fig, ax = plt.subplots(3,4, figsize=(10,5), num='scatter plots')
    #
    # ax[0,0].plot(above_HWL_dz,above_HWL_mgs, '.')
    # ax[0,1].plot(above_HWL_dz,above_HWL_dmgs, '.')
    # ax[0,2].plot(above_HWL_dz,above_HWL_last_mgs, '.')
    # ax[0,3].plot(above_HWL_dz_for_last_dz,above_HWL_last_dz, '.')

    ''' 2 things to accomplish: (all with longsshore transects only)
    - spatial correlation: correlate each tide, no mean removal
    - spatial correlation: correlate all tides (dz, dmgs only), remove mean
        - append mean-removed values to be correlated to new array. correalte at end of script
    - temporal correlation:  correlate means for each tide, no mean removal
        - append means to new array
        - correlate at end of script
    '''


saveFlag = 0
# export figs
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','MSD')

    # save_figures(savedn, 'correlation_spatialonly', fig1)
    save_figures(savedn, 'correlation_spatialonly_RCEM', fig01)
    # save_figures(savedn, 'dz_dmgs_scatter', fig2)
    # # # save_figures(savedn, 'delta_bed_level_swash_depth_scatter', fig02)
    # save_figures(savedn, 'bed_level_swash_depth_scatter_and_histogram_'+'tide'+tide+'_chunk'+chunk, fig02)
    # save_figures(savedn, 'delta_bed_level_histogram', fig03)
