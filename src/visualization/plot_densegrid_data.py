# -*- coding: utf-8 -*-
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

# use funtions from profile plotting script
from src.visualization.plot_beach_profile_data import *

# %matplotlib qt5

plt.close('all')


saveFlag = 0
saveCorr = 0

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux

figsdn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures')

grid_spec = "dense_array2"

start_tide = 14

tide_range = range(start_tide, 28)
tide_axis = np.arange(start_tide+1,28) # for plotting later

# guess at cross-shore profile length
npts = 50

hwl = [-21,-9,-15,-15,-18,-15,-15,-15,-18,-21,-18,-18,-18,-18]
hwl = hwl[1:]

start_rows = [0, 1, 2, 3, 4, 5]
ytrans = np.linspace(-15,-5,6)

fig12, ax12 = plt.subplots(2,1,figsize=(5,9), num='correlation against mean MGS')\

fig16, ax16 = plt.subplots(2,1,figsize=(5,9), num='correlation against mean sort')

fig13, ax13 = plt.subplots(2,1,figsize=(5,9), num='correlation against high tide elevation')

fig14, ax14 = plt.subplots(6,1,figsize=(5,9), num='correlation coefficients - all')

fig15, ax15 = plt.subplots(2,1,figsize=(5,9), num='correlation against HWL')



for start_row in start_rows:

    counter = 0

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
    mean_M0 = []
    mean_M1 = []


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

        mgs0 = np.array(jnk['mean_grain_size']).reshape(6, 24)
        sort0 = np.array(jnk['sorting']).reshape(6, 24)

        # foo = np.load(gpsfn, allow_pickle=True).item()
        with open(gpsfn, 'r') as fpp:
            foo = json.load(fpp)

        # reshape in to dense array dims
        x0 = np.array(foo['x']).reshape(6, 24)
        y0 = np.array(foo['y']).reshape(6, 24)
        z0 = np.array(foo['z']).reshape(6, 24)

        mgs = np.pad(mgs0[start_row,:], (0, npts-len(mgs0[start_row,:])), 'constant', constant_values=(np.nan,np.nan))
        sort = np.pad(sort0[start_row,:], (0, npts-len(sort0[start_row,:])), 'constant', constant_values=(np.nan,np.nan))

        # added mean MSG array
        mean_M0.append(np.nanmean(mgs))
        mean_M1.append(np.nanmean(sort))

        x = np.pad(x0[start_row,:], (0, npts-len(x0[start_row,:])), 'constant', constant_values=(np.nan,np.nan))
        y = np.pad(y0[start_row,:], (0, npts-len(y0[start_row,:])), 'constant', constant_values=(np.nan,np.nan))
        z = np.pad(z0[start_row,:], (0, npts-len(z0[start_row,:])), 'constant', constant_values=(np.nan,np.nan))

        # for first iteration
        if counter == 0:

            last_z = z
            dz = z - last_z

            last_mgs = mgs
            last_sort = sort
            dmgs = mgs - last_mgs
            dsort = sort - last_sort

            plt.figure(num='base profile')
            plt.plot(x,z)
            plt.ylabel('z [m]')
            plt.xlabel('x [m]')

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

            ## cross-correlation plots
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
            #     savedn = os.path.join(figsdn,'beach_profile',grid_spec,'line'+str(start_row),'cross_correlation',tide)
            #     savefn = 'dz_' + plt_tag
            #
            #     save_figures(savedn, savefn, fig)


        #tide 19: +1 more DGS obs than GPS
        ax01[0].plot(y0, z0)
        ax01[1].plot(y0, dz[:len(y0)])
        ax01[2].plot(y0, mgs0[:len(y0)])
        ax01[3].plot(y0, sort0[:len(y0)])

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


    # plt.figure(100)
    # plt.imshow(dz_line, cmap='bwr', vmin=-0.35, vmax=0.35)
    # plt.colorbar()
    #
    # plt.figure(101)
    # plt.imshow(mgs_line, cmap='inferno')
    # plt.colorbar()
    #
    # plt.figure(102)
    # plt.imshow(sort_line, cmap='inferno')
    # plt.colorbar()

    fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(3,6), num='cumulative change')
    ax20 = ax2[0].imshow(sum_dz_line[:maxreal,:], cmap='bwr', vmin=-0.35, vmax=0.35, \
    extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    clb20 = fig2.colorbar(ax20, ax=ax2[0])
    clb20.ax.set_title('dz [m]')
    ax21 = ax2[1].imshow(sum_dmgs_line[:maxreal,:], cmap='bwr', vmin=-30, vmax=30, \
    extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    clb21 = fig2.colorbar(ax21, ax=ax2[1])
    clb21.ax.set_title('dmgs [mm]')
    ax22 = ax2[2].imshow(sum_dsort_line[:maxreal,:], cmap='bwr', vmin=-25, vmax=25, \
    extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    clb22 = fig2.colorbar(ax22, ax=ax2[2])
    ax2[2].set_ylabel('cross-shore [m]')
    ax2[2].set_xlabel('tide')
    clb22.ax.set_title('dsort [mm]')
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(3,6), num='tide-tide change change')
    ax30 = ax3[0].imshow(dz_line[:maxreal,:], cmap='bwr', vmin=-0.2, vmax=0.2, \
    extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    clb30 = fig3.colorbar(ax30, ax=ax3[0])
    clb30.ax.set_title('dz [m]')
    ax31 = ax3[1].imshow(dmgs_line[:maxreal,:], cmap='bwr', vmin=-20, vmax=20, \
    extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    clb31 = fig3.colorbar(ax31, ax=ax3[1])
    clb31.ax.set_title('dmgs [mm]')
    ax32 = ax3[2].imshow(dsort_line[:maxreal,:], cmap='bwr', vmin=-15, vmax=15, \
    extent=[tide_range[1],tide_range[-1],15,-30], aspect='auto')
    clb32 = fig3.colorbar(ax32, ax=ax3[2])
    ax3[2].set_ylabel('cross-shore [m]')
    ax3[2].set_xlabel('tide')
    clb32.ax.set_title('dsort [mm]')
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
    # fig13, ax13 = plt.subplots(2,1,figsize=(5,9), num='correlation against high tide elevation')
    ax13[0].plot(maxdepth[1:], corrcoeffs_dz_mgs, '.')
    ax13[1].plot(maxdepth[1:], corrcoeffs_dz_dmgs, '.')

    ax15[0].plot(hwl - ytrans[start_row], corrcoeffs_dz_mgs, '.')
    ax15[1].plot(hwl - ytrans[start_row], corrcoeffs_dz_dmgs, '.')

    # plot correlation coefficients against high tide elevation
    # plot changes in profile (z, mgs, ...) against hydrodynamics
    # fig12, ax12 = plt.subplots(2,1,figsize=(5,9), num='correlation against mean MGS')
    ax12[0].plot(mean_M0[1:], corrcoeffs_dz_mgs, '.')
    ax12[1].plot(mean_M0[1:], corrcoeffs_dz_dmgs, '.')

    ax16[0].plot(mean_M1[1:], corrcoeffs_dz_mgs, '.')
    ax16[1].plot(mean_M1[1:], corrcoeffs_dz_dmgs, '.')


    # plot corr coeffs for each transect
    # fig9, ax9 = plt.subplots(4,1,figsize=(5,9), num='correlation coefficients - all')
    ax14[0].plot([-0.5,5.5],[0,0],'k--')
    ax14[0].plot(start_row, mean_cc_dz_mgs, 'C0.')
    ax14[0].plot([start_row,start_row], [mean_cc_dz_mgs-std_cc_dz_mgs,mean_cc_dz_mgs+std_cc_dz_mgs], 'C0-')
    ax14[1].plot([-0.5,5.5],[0,0],'k--')
    ax14[1].plot(start_row, mean_cc_dz_sort, 'C0.')
    ax14[1].plot([start_row,start_row], [mean_cc_dz_sort-std_cc_dz_sort,mean_cc_dz_sort+std_cc_dz_sort], 'C0-')
    ax14[2].plot([-0.5,5.5],[0,0],'k--')
    ax14[2].plot(start_row, mean_cc_dz_dmgs,'C0.')
    ax14[2].plot([start_row,start_row], [mean_cc_dz_dmgs-std_cc_dz_dmgs,mean_cc_dz_dmgs+std_cc_dz_dmgs], 'C0-')
    ax14[3].plot([-0.5,5.5],[0,0],'k--')
    ax14[3].plot(start_row, mean_cc_dz_dsort, 'C0.')
    ax14[3].plot([start_row,start_row], [mean_cc_dz_dsort-std_cc_dz_dsort,mean_cc_dz_dsort+std_cc_dz_dsort], 'C0-')
    ax14[4].plot([-0.5,5.5],[0,0],'k--')
    ax14[4].plot(start_row, mean_cc_dz_last_mgs,'C0.')
    ax14[4].plot([start_row,start_row], [mean_cc_dz_last_mgs-std_cc_dz_last_mgs,mean_cc_dz_last_mgs+std_cc_dz_last_mgs], 'C0-')
    ax14[5].plot([-0.5,5.5],[0,0],'k--')
    ax14[5].plot(start_row, mean_cc_dz_last_sort, 'C0.')
    ax14[5].plot([start_row,start_row], [mean_cc_dz_last_sort-std_cc_dz_last_sort,mean_cc_dz_last_sort+std_cc_dz_last_sort],'C0-')


ax14[0].autoscale(enable=True, axis='x', tight=True)
ax14[0].autoscale(enable=True, axis='y', tight=True)
ax14[1].autoscale(enable=True, axis='x', tight=True)
ax14[1].autoscale(enable=True, axis='y', tight=True)
ax14[2].autoscale(enable=True, axis='x', tight=True)
ax14[2].autoscale(enable=True, axis='y', tight=True)
ax14[3].autoscale(enable=True, axis='x', tight=True)
ax14[3].autoscale(enable=True, axis='y', tight=True)
ax14[4].autoscale(enable=True, axis='x', tight=True)
ax14[4].autoscale(enable=True, axis='y', tight=True)
ax14[5].autoscale(enable=True, axis='x', tight=True)
ax14[5].autoscale(enable=True, axis='y', tight=True)
fig14.tight_layout()



ax12[0].set_ylabel(r'$R^2 (\Delta z,M_0)$')
ax12[1].set_ylabel(r'$R^2 (\Delta z,\Delta M_0)$')
ax12[1].set_xlabel('mean MGS [mm]')
fig12.tight_layout()

ax16[0].set_ylabel(r'$R^2 (\Delta z,M_1)$')
ax16[1].set_ylabel(r'$R^2 (\Delta z,\Delta M_1)$')
ax16[1].set_xlabel('mean sort. [mm]')
fig16.tight_layout()

ax13[0].set_ylabel(r'$R^2 (\Delta z,M_0)$')
ax13[1].set_ylabel(r'$R^2 (\Delta z,\Delta M_0)$')
ax13[1].set_xlabel('max depth [m]')
fig13.tight_layout()

ax15[0].set_ylabel(r'$R^2 (\Delta z,M_0)$')
ax15[1].set_ylabel(r'$R^2 (\Delta z,\Delta M_0)$')
ax15[1].set_xlabel('HWL [m]')
fig15.tight_layout()


# EXPORT PLOTS
if saveFlag == 1:

    savedn = os.path.join(figsdn,'beach_profile',grid_spec,'line'+str(start_row))

    save_figures(savedn, 'elevation_and_grainsize', fig01)
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
