# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:00 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

%matplotlib qt5


# for portability
homechar = "C:\\"

validation_dns = ["LabValidation_xpos05", "OutdoorValidation_xpos05"]
# validation_dns = ["LabValidation_xmin05", "OutdoorValidation_xmin05"]
# validation_dns = ["LabValidation_xmin05", "OutdoorValidation_xmin05_ms5p5"]
# validation_dns = ["LabValidation_xmin05", "OutdoorValidation_xmin05_3pcwindowing"]
validation_dns = ["OutdoorValidation_xmin05", "OutdoorValidation_xpos05"]


# sample date and location
date_str0 = ["Oct21", "Oct21", "Oct21", "Oct21", "Oct21", \
             "Oct21", "Oct25", "Oct25", "Oct25", "Oct25"]
tide0 = ["_bay1", "_bay2", "_bay3", "_horn1", "_horn2", "_horn3", \
         "_bay1", "_bay2", "_horn1", "_horn2"]

### INITIALIZE FIGURES
# mean and standard dev of gsdist compared
fig1, ax1 = plt.subplots(nrows=2, ncols=1, num="compare mean and stdev")

old_dgs = []
new_dgs = []

for n in range(0, 2):

    validation_dn = validation_dns[n]

    # load in lookup tables
    # for lab validation

    if n is 3:
        lookupdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "references", \
                              "img2sample_lookup_table.csv")
    else:
        # for outdoor validation
        lookupdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "references", \
                              "img2sample_lookup_table_outdoorvalidation.csv")

    # sample index
    sample_loc = range(0, len(tide0))

    imgnames = []
    dates = []
    locs = []
    depths = []
    iters = []

    with open(lookupdir, 'rt') as f:
        lookup = csv.reader(f, delimiter=',')

        for row in lookup:
            imgnames.append(row[0])
            dates.append(row[1])
            locs.append(row[2])
            depths.append(row[3])
            iters.append(row[4])

    rms_mgs_vec = []
    rms_std_vec = []




    for ii in sample_loc:


        gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize_dists", validation_dn, date_str0[ii] + tide0[ii])

        allfiles = os.listdir(gsizedir)

        for jj in range(0, len(allfiles)):


            dgsjnk = np.load(os.path.join(gsizedir, allfiles[jj]), encoding='latin1').item()

            imgnum = str(int(allfiles[jj][4:8])) # removes leading zeros

            Iimg = imgnames.index(imgnum)

            # load associated sieve data
            bar = np.load(os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                                       "interim", "grainsize_dists", "sieved", \
                                       dates[Iimg][3:6] + dates[Iimg][:2] + '_' + \
                                       locs[Iimg] + '_' + depths[Iimg] + '.npy')).item()

            gsfreqs = bar['grain size frequencies']#[2:-1] # omitting 45 mm size
            gsbins = bar['grain size bins']#[2:-1]
            cumsum = bar['cumulative sum']
            # rescale if necessary
    #            binsum = np.sum(gsfreqs)
    #            gsfreqs = gsfreqs/binsum

            # resample dist
            xp_stt = np.min(gsbins)
            xp_end = np.max(gsbins)
            xp = np.linspace(xp_stt, xp_end, 10000)

            cumsum_interp = np.interp(xp, np.flipud(gsbins),  np.flipud(cumsum))
            p05 = np.argmin(np.abs(0.05 - cumsum_interp))
            p10 = np.argmin(np.abs(0.10 - cumsum_interp))
            p16 = np.argmin(np.abs(0.16 - cumsum_interp))
            p25 = np.argmin(np.abs(0.25 - cumsum_interp))
            p50 = np.argmin(np.abs(0.50 - cumsum_interp))
            p75 = np.argmin(np.abs(0.75 - cumsum_interp))
            p84 = np.argmin(np.abs(0.84 - cumsum_interp))
            p90 = np.argmin(np.abs(0.90 - cumsum_interp))
            p95 = np.argmin(np.abs(0.95 - cumsum_interp))

            mean_gs = np.sum(gsfreqs*gsbins)
            std_gs = np.sqrt(np.sum(gsfreqs*(gsbins - mean_gs)**2))
            skew_gs = (np.sum(gsfreqs*(gsbins - mean_gs)**3))/(100*std_gs**3)
            kurt_gs = (np.sum(gsfreqs*(gsbins - mean_gs)**4))/(100*std_gs**4)

            perc_freqs = dgsjnk['percentile_values'][4]

            cumsum_img = np.cumsum(dgsjnk['grain size frequencies'])

            rms_mgs_vec.append(mean_gs - dgsjnk['mean grain size'])
            rms_std_vec.append(std_gs - dgsjnk['grain size sorting'])


            # ### FIGURES
            #
            # # mean and standard dev of gsdist compared
            # fig1, ax1 = plt.subplots(nrows=2, ncols=1, num="compare mean and stdev")
            if n is 0:
                l1, = ax1[0].plot(mean_gs, dgsjnk['mean grain size'], 'k.', label='no window')
            else:
                l2, = ax1[0].plot(mean_gs, dgsjnk['mean grain size'], 'r.', label='window')
            # plt.plot(mean_gs, dgsjnk['mean grain size'], 'k.')
    #            plt.plot(mean_gs, perc_freqs, 'c.')
            # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
            ax1[0].plot(np.arange(30), np.arange(30))
            ax1[0].set_ylabel('DGS mean [mm]')
            if n is 0:
                ax1[1].plot(std_gs, dgsjnk['grain size sorting'], 'k.')
            else:
                # ax1[1].plot(std_gs, dgsjnk['grain size sorting'], 'r.')
                ax1[1].plot(std_gs, np.mean(dgsjnk['grain size sorting']), 'r.')
            ax1[1].plot(np.arange(20), np.arange(20))
            ax1[1].set_xlabel('sieve [mm]')
            ax1[1].set_ylabel('DGS sorting [mm]')
            # ax1[1].title('sorting. RMSE = ' + str(rmse_sort))
            fig1.tight_layout()
            # plt.subplot(223)
            # plt.plot(skew_gs, dgsjnk['grain size skewness'], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('skewness')
            # plt.xlabel('sieved')
            # plt.ylabel('DGS')
            # plt.subplot(224)
            # plt.plot(kurt_gs, dgsjnk['grain size kurtosis'], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('kurtosis')

            if n is 0:
                old_dgs.append(dgsjnk['mean grain size'])
            elif n is 1:
                new_dgs.append(dgsjnk['mean grain size'])

            # plt.figure(102)
            # plt.subplot(231)
            # plt.plot(xp[p10], dgsjnk['percentile_values'][1], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('10th percentile')
            # plt.subplot(232)
            # plt.plot(xp[p16], dgsjnk['percentile_values'][2], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('16th percentile')
            # plt.subplot(233)
            # plt.plot(xp[p25], dgsjnk['percentile_values'][3], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('25th percentile')
            # plt.subplot(234)
            # plt.plot(xp[p50], dgsjnk['percentile_values'][4], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('50th percentile')
            # plt.subplot(235)
            # plt.plot(xp[p75], dgsjnk['percentile_values'][5], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('75th percentile')
            # plt.subplot(236)
            # plt.plot(xp[p84], dgsjnk['percentile_values'][6], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('84th percentile')
            #
            # plt.figure(103)
            # plt.plot(xp[p50], dgsjnk['percentile_values'][4], 'm.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.xlabel('d50')
    #
    #        plt.figure(1000+ii)
    #        plt.plot(xp, 1-cumsum_interp, 'k.')
    #        plt.plot(jnk['grain size bins'], cumsum_img)


    #        plt.figure(ii)
    #        plt.plot(jnk['grain size bins'], jnk['grain size frequencies'])
    #        plt.plot(gsbins, gsfreqs)
    #        plt.plot(mean_gs, 0, 'ro')

    #        plt.figure(9999)
    #        plt.plot(gsbins,  bar['cumulative sum'])
    #        plt.plot(xp, cumsum_interp,'r')

    #        plt.figure(1000+ii)
    #        plt.plot(jnk['percentile_values'], jnk['percentiles'])
    #        plt.plot(gsbins,  bar['cumulative sum'])
    #        plt.xlabel('values')
    #        plt.ylabel('percentiles')

    #        plt.plot(gsbins, gsfreqs)
    #        plt.plot(mean_gs, 0, 'ro')

    #    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,7))
    #    pos1 = ax1.imshow(gsize['mean_grain_size'], cmap='inferno')
    #    fig.colorbar(pos1, ax=ax1)
    #    pos2 = ax2.imshow(gsize['sorting'], cmap='inferno')
    #    fig.colorbar(pos2, ax=ax2)

    rmse_mgs = np.sqrt(np.nanmean(np.array(rms_mgs_vec)**2))
    rmse_sort = np.sqrt(np.nanmean(np.array(rms_std_vec)**2))

# ax1[0].legend(['no windowing', 'windowing'], loc="upper left")
ax1[0].legend([l1, l2],['x=0.5; lab validation', 'x=0.5; outdoor validation'], loc="upper left")

# ax1.plot(xtr, color='r', label='HHZ 1')
# ax1.legend(loc="upper right")
# ax2.plot(xtr, color='r', label='HHN')
# ax2.legend(loc="upper right")
# ax3.plot(xtr, color='r', label='HHE') 
# ax3.legend(loc="upper right")

fig4, ax4 = plt.subplots(nrows=1, ncols=1, num='1-1')
ax4.plot(old_dgs, new_dgs, 'k.', label='no window')
ax4.plot(np.arange(30), np.arange(30))
ax4.set_ylabel('DGS mean [mm]')
fig1.tight_layout()



saveFlag = 0
if saveFlag == 1:
    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','dgs_validation')
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig1.savefig(os.path.join(savedn, 'dgs_vs_sieve_xpos05_in-out.png'), dpi=1000, transparent=True)
    fig1.savefig(os.path.join(savedn, 'dgs_vs_sieve_xpos05_in-out.pdf'), dpi=None, transparent=True)
