#!/usr/bin/env python3
"""
Created on Thu Nov 22 15:47:31 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import stats
from scipy.signal import correlate
import json

plt.close('all')

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


def main():
    saveFlag = 0
    saveCorr = 0

    # for portability
    # homechar = "C:\\"
    homechar = os.path.expanduser("~") # linux

    figsdn = os.path.join(homechar,'Projects','AdvocateBeach2018',\
    'reports','figures')

    grid_spec = "cross_shore"
    # grid_spec = "longshore2"
    # grid_spec = "longshore1"

    if grid_spec == 'cross_shore':
        start_tide = 13
    elif grid_spec == 'longshore1':
        start_tide = 15
    else:
        start_tide = 14

    tide_range = range(start_tide, 28)
    tide_axis = np.arange(start_tide+1,28) # for plotting later

    counter = 0

    # guess at cross-shore profile length
    npts = 50

    sum_dz_line = np.zeros((npts,len(tide_range)-1))
    dz_line = np.zeros((npts,len(tide_range)-1))
    z_line = np.zeros((npts,len(tide_range)-1))
    sum_dz = np.zeros((npts,))

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

    Hs = []
    Tp = []
    steepness = []
    iribarren = []
    wave_energy = []
    wave_energy_wind = []
    wave_energy_swell = []
    maxdepth = []


    fig01, ax01 = plt.subplots(nrows=4,ncols=1, num='profiles', sharex=True)

    for ii in tide_range:

        tide = "tide" + str(ii)

        gsizefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                              "grainsize", "beach_surveys", tide, grid_spec + ".json")

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

            last_mgs = mgs
            last_sort = sort
            dmgs = mgs - last_mgs
            dsort = sort - last_sort

            plt.figure(num='base profile')
            plt.plot(y,z)
            plt.ylabel('z [m]')
            plt.xlabel('y [m]')

        else:

            dz = z - last_z
    #        last_z = z

            last_mgs0 = last_mgs
            last_sort0 = last_sort

            dmgs = mgs - last_mgs
            dsort = sort - last_sort

            # update
            last_z = z
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

            # undifferenced
            z_line[:,counter-1] = z
            mgs_line[:,counter-1] = mgs
            sort_line[:,counter-1] = sort

            # correlate dz with mgs(t-1), mgs, dmgs

            tmp_dz = dz
            tmp_dz = tmp_dz[~np.isnan(tmp_dz)] - np.nanmean(dz)

            tmp_mgs = mgs
            tmp_dmgs = dmgs
            tmp_sort = sort
            tmp_dsort = dsort
            tmp_last_mgs = last_mgs0
            tmp_last_sort = last_sort0

            plt_tag = 'mgs'
            tmp_mgs = tmp_mgs[~np.isnan(tmp_mgs)] - np.nanmean(mgs)

            corrcoeffs_dz_mgs.append(np.corrcoef(tmp_dz,tmp_mgs[:len(tmp_dz)])[0,1])
            corrcoeffs_dz_dmgs.append(np.corrcoef(tmp_dz,tmp_dmgs[:len(tmp_dz)])[0,1])
            corrcoeffs_dz_sort.append(np.corrcoef(tmp_dz,tmp_sort[:len(tmp_dz)])[0,1])
            corrcoeffs_dz_dsort.append(np.corrcoef(tmp_dz,tmp_dsort[:len(tmp_dz)])[0,1])
            corrcoeffs_dz_last_mgs.append(np.corrcoef(tmp_dz,tmp_last_mgs[:len(tmp_dz)])[0,1])
            corrcoeffs_dz_last_sort.append(np.corrcoef(tmp_dz,tmp_last_sort[:len(tmp_dz)])[0,1])

    # plt.figure()
    # plt.plot(tmp_dz)
    # plt.figure()
    # plt.plot(dmgs)
            fig, ax1 = plt.subplots(2, 1, sharex=True, num='tide'+str(ii))
            ax1[0].xcorr(tmp_dz, tmp_mgs[:len(tmp_dz)], usevlines=True, maxlags=10, normed=True, lw=2)
            ax1[0].set_ylabel('correlation ($\Delta z, M_0$)')

            ax1[1].xcorr(tmp_dz, tmp_dmgs[:len(tmp_dz)], usevlines=True, maxlags=10, normed=True, lw=2)
            ax1[1].set_ylabel('correlation ($\Delta z, \Delta M_0$)')
            ax1[1].set_xlabel('lag')

            # EXPORT PLOTS
            if saveCorr == 1:
                savedn = os.path.join(figsdn,'beach_profile',grid_spec,'cross_correlation',tide)
                savefn = 'dz_' + plt_tag

                save_figures(savedn, savefn, fig)


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

    fig9, ax9 = plt.subplots(2,1,figsize=(3,7), gridspec_kw={'height_ratios': [3, 1]}, num='correlation coefficients')
    ax9[0].plot([0,0],[13,28],'k--')
    ax9[0].plot(corrcoeffs_dz_mgs, tide_axis, 'C0.')
    ax9[0].plot(corrcoeffs_dz_sort, tide_axis, 'C1.')
    ax9[0].plot(corrcoeffs_dz_dmgs, tide_axis, 'C2.')
    ax9[0].plot(corrcoeffs_dz_dsort, tide_axis, 'C3.')
    ax9[0].plot(corrcoeffs_dz_last_mgs, tide_axis, 'C4.')
    ax9[0].plot(corrcoeffs_dz_last_sort, tide_axis, 'C5.')
    ax9[0].autoscale(enable=True, axis='y', tight=True)
    ax9[0].invert_yaxis()
    ax9[0].set_ylabel('tide')
    ax90xlim = ax9[0].get_xlim()

    ax9[1].plot([0,0],[-0.5,5.5],'k--')
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
    ax9[1].autoscale(enable=True, axis='y', tight=True)
    ax9[1].invert_yaxis()
    ax9[1].set_xlabel('correlation coefficient')
    setstr = [r'$\Delta z,M_0$', r'$\Delta z,M_1$', r'$\Delta z,\Delta M_0$',\
    r'$\Delta z,\Delta M_1$', r'$\Delta z,M_0[t-1]$', r'$\Delta z,M_1[t-1]$']
    setind = [0, 1, 2, 3, 4, 5]
    ax9[1].set_yticks(setind)
    ax9[1].set_yticklabels(setstr)
    ax9[1].set_xlim(ax90xlim)
    fig9.tight_layout()

    # plot changes in profile (z, mgs, ...) against hydrodynamics
    fig10, ax10 = plt.subplots(3,1,figsize=(5,9), num='profile change')
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
    ax11[0].plot(tide_axis, Hs[1:], '.')
    ax11[0].set_ylabel('$H_s$ [m]')
    ax11[1].plot(tide_axis, np.nanmean(mgs_line,axis=0), '.')
    # ax11[1].errorbar(tide_axis, np.nanmean(mgs_line,axis=0),
    #             xerr=0,
    #             yerr=np.nanmean(sort_line,axis=0))
    ax11[1].set_ylabel('mean grain size [mm]')
    ax11[1].set_xlabel('tide')
    fig11.tight_layout()


    # plot correlation coefficients against high tide elevation
    # plot changes in profile (z, mgs, ...) against hydrodynamics
    fig12, ax12 = plt.subplots(3,1,figsize=(5,9), num='correlation against high tide elevation')
    ax12[0].plot(maxdepth[1:], corrcoeffs_dz_mgs, '.')
    ax12[0].set_ylabel(r'$R^2 (\Delta z,M_0)$')
    ax12[1].plot(maxdepth[1:], corrcoeffs_dz_dmgs, '.')
    ax12[1].set_ylabel(r'$R^2 (\Delta z,\Delta M_0)$')
    ax12[1].set_xlabel('max depth [m]')
    fig12.tight_layout()




    # EXPORT PLOTS
    if saveFlag == 1:

        savedn = os.path.join(figsdn,'beach_profile',grid_spec)

        save_figures(savedn, 'elevation_and_grainsize', fig01)
        save_figures(savedn, 'cumulative_elevation_and_grainsize', fig1)
        save_figures(savedn, 'cumulative_elevation_and_grainsize_change', fig2)
        save_figures(savedn, 'tidal_elevation_and_grainsize_change', fig3)
        save_figures(savedn, 'Hs_corr_coeff', fig4)
        save_figures(savedn, 'energy_corr_coeff', fig5)
        save_figures(savedn, 'Tp_corr_coeff', fig6)
        save_figures(savedn, 'steepness_corr_coeff', fig7)
        save_figures(savedn, 'iribarren_corr_coeff', fig8)
        save_figures(savedn, 'pearson_correlation_coefficients', fig9)
        save_figures(savedn, 'profile_change', fig10)
        save_figures(savedn, 'grain_size_and_waveheight_timeseries', fig11)


if __name__ == '__main__':
    main()
