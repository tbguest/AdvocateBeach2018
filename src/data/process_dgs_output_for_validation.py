# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:00 2018

@author: Owner
"""


%reset -f
%matplotlib qt5

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math


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


# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux


# validation_dns = ["LabValidation_xpos05", "OutdoorValidation_xpos05"]
# validation_dns = ["LabValidation_xmin05", "OutdoorValidation_xmin05"]
# validation_dns = ["LabValidation_xmin05", "OutdoorValidation_xmin05_ms5p5"]
# validation_dns = ["LabValidation_xmin05", "OutdoorValidation_xmin05_3pcwindowing"]

# validation_dns = ["OutdoorValidation_xmin05", "OutdoorValidation_xpos05"]
# validation_dns = ["OutdoorValidation_x0", "OutdoorValidation_xpos05"]
# validation_dns = ["OutdoorValidation_x08", "OutdoorValidation_x09"]
# validation_dns = ["OutdoorValidation_x0", "OutdoorValidation_x0_maxscale5"]

# validation_dns = ["OutdoorValidation_x05", "OutdoorValidation_x08", "OutdoorValidation_x15"]

validation_dns = ["OutdoorValidation_maxscale48_x05", "OutdoorValidation_maxscale48_x08", "OutdoorValidation_maxscale48_x15"]


# sample date and location
date_str0 = ["Oct21", "Oct21", "Oct21", "Oct21", "Oct21", \
             "Oct21", "Oct25", "Oct25", "Oct25", "Oct25"]
tide0 = ["_bay1", "_bay2", "_bay3", "_horn1", "_horn2", "_horn3", \
         "_bay1", "_bay2", "_horn1", "_horn2"]

### INITIALIZE FIGURES
# # mean and standard dev of gsdist compared
# fig1, ax1 = plt.subplots(nrows=2, ncols=2, num="compare mean and stdev")
# fig2, ax2 = plt.subplots(nrows=2, ncols=2, num="compare mean with pt count")
# fig3, ax3 = plt.subplots(nrows=2, ncols=2, num="compare sieve with pt count")
# fig11, ax11 = plt.subplots(nrows=2, ncols=2, num="compare sort arith, geom, etc")

# mean and standard dev of gsdist compared
fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), num="compare mean and stdev - sieve")
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), num="compare mean and stdev - pt count")
fig3, ax3 = plt.subplots(nrows=2, ncols=2, num="compare sieve with pt count")
fig11, ax11 = plt.subplots(nrows=2, ncols=2, num="compare sort arith, geom, etc")

rmse_mgs_sieve_dgs = []
rmse_mgs_count_dgs = []


old_dgs = []
new_dgs = []

for n in range(0, 3):

    validation_dn = validation_dns[n]

    # load in lookup tables
    # for lab validation

    if n is 4:
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

    rms_mgs_vec_dgs_sieve = []
    rms_mgs_vec_dgs_count = []
    rms_std_vec = []




    for ii in sample_loc:


        gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize", "validation",validation_dn, date_str0[ii] + tide0[ii])

        allfiles = sorted(os.listdir(gsizedir))

        ### added to compare point counts ####
        pointcountdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize", "validation",'point-counts')
        ###############

        for jj in range(0, len(allfiles)):

            dgsjnk = np.load(os.path.join(gsizedir, allfiles[jj]), encoding='latin1', allow_pickle=True).item()

            gsfreq = dgsjnk['grain size frequencies']

            logmgs = []
            for elem in dgsjnk['grain size bins']:
                logmgs.append(-math.log2(elem))



            ### added to compare point counts ####
            inum = str(int(allfiles[jj][4:8]))

            Igarbo = imgnames.index(inum)

            # if exists, compute gsdir from point count
            if os.path.exists(os.path.join(pointcountdir,  date_str0[ii] + tide0[ii] + '_' + depths[Igarbo] + '.npy')):

                ptcount = np.load(os.path.join(pointcountdir,  date_str0[ii] + tide0[ii] + '_' + depths[Igarbo] + \
                    '.npy'), allow_pickle=True)

                dens0, wghts = np.histogram(ptcount,np.arange(1,100,2))
                dens1 = dens0/np.sum(dens0)
                if len(dens1) < len(wghts):
                    padlen = len(wghts) - len(dens1)
                    dens = np.pad(dens1, (0,padlen), 'constant', constant_values=0)

                # arithmetic (point count)
                mgs_a_ptc = np.sum(dens*wghts)
                sort_a_ptc = np.sqrt(np.sum(dens*(wghts - mgs_a_ptc)**2))
                skew_a_ptc = np.sum(dens*(wghts - mgs_a_ptc)**3)/sort_a_ptc**3
                kurt_a_ptc = np.sum(dens*(wghts - mgs_a_ptc)**4)/sort_a_ptc**4

                # geometric (point count)
                mgs_g_ptc = np.exp(np.sum(dens*np.log(wghts)))
                sort_g_ptc = np.exp(np.sqrt(np.sum(dens*(np.log(wghts) - np.log(mgs_g_ptc))**2)))

                ###

            ###############



# plt.figure(6)
# plt.plot(dgsjnk['percentile_values'],dgsjnk['percentiles'])
#
# plt.figure(7)
# plt.plot(dgsjnk['grain size bins'],dgsjnk['grain size frequencies'])

            ### added

            d50_dgs = dgsjnk['percentile_values'][4]

            # arithmetic
            mgs_a = np.sum(gsfreq*dgsjnk['grain size bins'])
            sort_a = np.sqrt(np.sum(gsfreq*(dgsjnk['grain size bins'] - mgs_a)**2))
            skew_a = np.sum(gsfreq*(dgsjnk['grain size bins'] - mgs_a)**3)/sort_a**3
            kurt_a = np.sum(gsfreq*(dgsjnk['grain size bins'] - mgs_a)**4)/sort_a**4

            # geometric
            mgs_g = np.exp(np.sum(gsfreq*np.log(dgsjnk['grain size bins'])))
            sort_g = np.exp(np.sqrt(np.sum(gsfreq*(np.log(dgsjnk['grain size bins']) - np.log(mgs_g))**2)))
            skew_g = np.sum(gsfreq*(np.log(dgsjnk['grain size bins']) - np.log(mgs_g))**3)/np.log(sort_g)**3
            kurt_g = np.sum(gsfreq*(np.log(dgsjnk['grain size bins']) - np.log(mgs_g))**4)/np.log(sort_g)**4

            # logarithmic (phi-scaled)
            mgs_p = np.sum(gsfreq*logmgs)
            sort_p = np.sqrt(np.sum(gsfreq*(logmgs - mgs_p)**2))
            skew_p = np.sum(gsfreq*(logmgs - mgs_p)**3)/sort_p**3
            kurt_p = np.sum(gsfreq*(logmgs - mgs_p)**4)/sort_p**4
            ###




            imgnum = str(int(allfiles[jj][4:8])) # removes leading zeros

            Iimg = imgnames.index(imgnum)

            # load associated sieve data
            bar = np.load(os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                                       "interim", "grainsize", "sieved", \
                                       dates[Iimg][3:6] + dates[Iimg][:2] + '_' + \
                                       locs[Iimg] + '_' + depths[Iimg] + '.npy'), allow_pickle=True).item()

            gsfreq_sieve = bar['grain size frequencies'][1:]/np.sum(bar['grain size frequencies'][1:])#[2:-1] # omitting 45 mm size
            gsbins = bar['grain size bins'][1:]
            cumsum = bar['cumulative sum'][1:]
            # rescale if necessary
    #            binsum = np.sum(gsfreq_sieve)
    #            gsfreq_sieve = gsfreq_sieve/binsum

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

            d50_sieve = xp[p50]

            logmgs_sieve = []
            for elem in gsbins:
                logmgs_sieve.append(-math.log2(elem))


            ##
            # arithmetic
            mgs_a_sieve = np.sum(gsfreq_sieve*gsbins)
            sort_a_sieve = np.sqrt(np.sum(gsfreq_sieve*(gsbins - mgs_a_sieve)**2))
            skew_a_sieve = (np.sum(gsfreq_sieve*(gsbins - mgs_a_sieve)**3))/(sort_a_sieve**3)
            kurt_a_sieve = (np.sum(gsfreq_sieve*(gsbins - mgs_a_sieve)**4))/(sort_a_sieve**4)

            # geometric
            mgs_g_sieve = np.exp(np.sum(gsfreq_sieve*np.log(gsbins)))
            sort_g_sieve = np.exp(np.sqrt(np.sum(gsfreq_sieve*(np.log(gsbins) - np.log(mgs_g_sieve))**2)))
            skew_g_sieve = np.sum(gsfreq_sieve*(np.log(gsbins) - np.log(mgs_g_sieve))**3)/np.log(sort_g_sieve)**3
            kurt_g = np.sum(gsfreq_sieve*(np.log(gsbins) - np.log(mgs_g_sieve))**4)/np.log(sort_g_sieve)**4

            # logarithmic (phi-scaled)
            mgs_p_sieve = np.sum(gsfreq_sieve*logmgs_sieve)
            sort_p_sieve = np.sqrt(np.sum(gsfreq_sieve*(logmgs_sieve - mgs_p_sieve)**2))
            skew_p_sieve = np.sum(gsfreq_sieve*(logmgs_sieve - mgs_p_sieve)**3)/sort_p_sieve**3
            kurt_p_sieve = np.sum(gsfreq_sieve*(logmgs_sieve - mgs_p_sieve)**4)/sort_p_sieve**4
            ###
            ##


            perc_freqs = dgsjnk['percentile_values'][4]

            cumsum_img = np.cumsum(dgsjnk['grain size frequencies'])

            rms_mgs_vec_dgs_sieve.append(mgs_a_sieve - mgs_a)
            if os.path.exists(os.path.join(pointcountdir,  date_str0[ii] + tide0[ii] + '_' + depths[Igarbo] + '.npy')):
                rms_mgs_vec_dgs_count.append(mgs_a_ptc - mgs_a)
            rms_std_vec.append(sort_a_sieve - dgsjnk['grain size sorting'])




            # ### FIGURES
            #
            # # mean and standard dev of gsdist compared
            # fig1, ax1 = plt.subplots(nrows=2, ncols=1, num="compare mean and stdev")
            if n is 0:
                l0, = ax1[0].plot(mgs_a_sieve, mgs_a, 'C0.', markersize=5, alpha=0.5, label='l0')
            elif n is 1:
                l1, = ax1[0].plot(mgs_a_sieve, mgs_a, 'k.', markersize=8, label='no window')
            else:
                l2, = ax1[0].plot(mgs_a_sieve, mgs_a, 'C1.', markersize=5, alpha=0.5, label='window')
            # plt.plot(mgs_a_sieve, dgsjnk['mean grain size'], 'k.')
    #            plt.plot(mgs_a_sieve, perc_freqs, 'c.')
            # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
            ax1[0].plot(np.linspace(0,30,num=100), np.linspace(0,33,num=100),'k--', linewidth=0.5)
            ax1[0].set_ylabel('DGS MGS [mm]')
            ax1[0].set_xlabel('sieve MGS [mm]')
            ax1[0].set_xlim([0, 30])
            ax1[0].set_ylim([0, 33])
            ax1[0].autoscale(enable=True, axis='x', tight=True)
            ax1[0].autoscale(enable=True, axis='y', tight=True)
            if n is 0:
                ax1[1].plot(sort_a_sieve, sort_a, 'C0.', markersize=5, alpha=0.5)
            elif n is 1:
                ax1[1].plot(sort_a_sieve, sort_a, 'k.', markersize=8)
                # ax1[1,0].plot(mgs_g_sieve, mgs_g, 'k.')
            else:
                ax1[1].plot(sort_a_sieve, sort_a, 'C1.', markersize=5, alpha=0.5)
                # ax1[1,0].plot(mgs_g_sieve, mgs_g, 'r.')
            ax1[1].plot(np.linspace(0,15,num=100), np.linspace(0,25,num=100),'k--', linewidth=0.5)
            ax1[1].set_xlabel('sieve sorting [mm]')
            ax1[1].set_ylabel('DGS sorting [mm]')
            ax1[1].set_xlim([0, 15])
            ax1[1].set_ylim([0, 25])
            ax1[1].autoscale(enable=True, axis='x', tight=True)
            ax1[1].autoscale(enable=True, axis='y', tight=True)

            # if n is 0:
            #     ax1[1,0].plot(mgs_g_sieve, mgs_g, 'k.')
            #     # ax1[1].plot(sort_a_sieve, sort_g, 'k.')
            # else:
            #     ## ax1[1].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'r.')
            #     ax1[1,0].plot(mgs_g_sieve, mgs_g, 'r.')
            #     # ax1[1].plot(sort_a_sieve, sort_g, 'r.')
            # ax1[1,0].plot(np.arange(20), np.arange(20))
            # ax1[1,0].set_xlabel('sieve geom. MGS [mm]')
            # ax1[1,0].set_ylabel('DGS geom. MGS [mm]')
            #
            # if n is 0:
            #     ax1[1,1].plot(sort_g_sieve, sort_g, 'k.', markersize=12)
            #     # ax1[1].plot(sort_a_sieve, sort_g, 'k.')
            # else:
            #     ## ax1[1].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'r.')
            #     ax1[1,1].plot(sort_g_sieve, sort_g, 'r.')
            #     # ax1[1].plot(sort_a_sieve, sort_g, 'r.')
            # ax1[1,1].plot(np.arange(20), np.arange(20))
            # ax1[1,1].set_xlabel('sieve geom. sorting [mm]')
            # ax1[1,1].set_ylabel('DGS geom. sorting [mm]')

            # ax1[1].title('sorting. RMSE = ' + str(rmse_sort))
            fig1.tight_layout()
            # plt.subplot(223)
            # plt.plot(skew_a_sieve, dgsjnk['grain size skewness'], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('skewness')
            # plt.xlabel('sieved')
            # plt.ylabel('DGS')
            # plt.subplot(224)
            # plt.plot(kurt_a_sieve, dgsjnk['grain size kurtosis'], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('kurtosis')




            # # mean and standard dev of gsdist compared
            # fig1, ax1 = plt.subplots(nrows=2, ncols=1, num="compare mean and stdev")
            if n is 0:
                l11, = ax11[0,0].plot(sort_a_sieve, sort_a, 'k.', label='no window')
            else:
                l12, = ax11[0,0].plot(sort_a_sieve, sort_a, 'r.', label='window')
            # plt.plot(mgs_a_sieve, dgsjnk['mean grain size'], 'k.')
    #            plt.plot(mgs_a_sieve, perc_freqs, 'c.')
            # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
            ax11[0,0].plot(np.arange(30), np.arange(30))
            ax11[0,0].set_ylabel('DGS arith. sort [mm]')
            ax11[0,0].set_xlabel('sieve arith. sort [mm]')
            if n is 0:
                # ax1[1,0].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'k.')
                ax11[1,0].plot(sort_g_sieve, sort_g, 'k.')
            else:
                # ax1[1,0].plot(sort_a_sieve, np.mean(dgsjnk['grain size sorting']), 'r.')
                ax11[1,0].plot(sort_g_sieve, sort_g, 'r.')
            ax11[1,0].plot(np.arange(20), np.arange(20))
            ax11[1,0].set_xlabel('sieve geom. sort [mm]')
            ax11[1,0].set_ylabel('DGS geom. sort [mm]')

            if n is 0:
                ax11[0,1].plot(sort_p_sieve, sort_p, 'k.')
                # ax1[1].plot(sort_a_sieve, sort_g, 'k.')
            else:
                ## ax1[1].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'r.')
                ax11[0,1].plot(sort_p_sieve,sort_p, 'r.')
                # ax1[1].plot(sort_a_sieve, sort_g, 'r.')
            ax11[0,1].plot(np.arange(20), np.arange(20))
            ax11[0,1].set_xlabel('sieve phi sort')
            ax11[0,1].set_ylabel('DGS phi sort')

            if n is 0:
                ax11[1,1].plot(d50_sieve, d50_dgs, 'k.', markersize=12)
                # ax1[1].plot(sort_a_sieve, sort_g, 'k.')
            else:
                ## ax1[1].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'r.')
                ax11[1,1].plot(d50_sieve, d50_dgs, 'r.')
                # ax1[1].plot(sort_a_sieve, sort_g, 'r.')
            ax11[1,1].plot(np.arange(20), np.arange(20))
            ax11[1,1].set_xlabel('sieve d50 [mm]')
            ax11[1,1].set_ylabel('DGS d50 [mm]')

            # ax1[1].title('sorting. RMSE = ' + str(rmse_sort))
            fig11.tight_layout()
            # plt.subplot(223)
            # plt.plot(skew_a_sieve, dgsjnk['grain size skewness'], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('skewness')
            # plt.xlabel('sieved')
            # plt.ylabel('DGS')
            # plt.subplot(224)
            # plt.plot(kurt_a_sieve, dgsjnk['grain size kurtosis'], 'k.')
            # plt.plot(np.arange(20), np.arange(20))
            # plt.title('kurtosis')








            if os.path.exists(os.path.join(pointcountdir,  date_str0[ii] + tide0[ii] + '_' + depths[Igarbo] + '.npy')):
                if n is 0:
                    l00, = ax2[0].plot(mgs_a_ptc, mgs_a, 'C0.', markersize=5, alpha=0.5, label='l0')
                elif n is 1:
                    l01, = ax2[0].plot(mgs_a_ptc, mgs_a, 'k.', markersize=8, label='no window')
                else:
                    l02, = ax2[0].plot(mgs_a_ptc, mgs_a, 'C1.', markersize=5, alpha=0.5, label='window')
                # plt.plot(mgs_a_sieve, dgsjnk['mean grain size'], 'k.')
        #            plt.plot(mgs_a_sieve, perc_freqs, 'c.')
                # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
                ax2[0].plot(np.linspace(0,30,num=100), np.linspace(0,33,num=100), 'k--', linewidth=0.5)
                ax2[0].set_ylabel('DGS MGS [mm]')
                ax2[0].set_xlabel('point count MGS [mm]')
                ax2[0].set_xlim([0, 30])
                ax2[0].set_ylim([0, 33])
                ax2[0].autoscale(enable=True, axis='x', tight=True)
                ax2[0].autoscale(enable=True, axis='y', tight=True)
                if n is 0:
                    ax2[1].plot(sort_a_ptc, sort_a, 'C0.', alpha=0.5, markersize=5)
                elif n is 1:
                    # ax1[1,0].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'k.')
                    ax2[1].plot(sort_a_ptc, sort_a, 'k.', markersize=8)
                else:
                    # ax1[1,0].plot(sort_a_sieve, np.mean(dgsjnk['grain size sorting']), 'r.')
                    ax2[1].plot(sort_a_ptc, sort_a, 'C1.', alpha=0.5, markersize=5)
                ax2[1].plot(np.linspace(0,13,num=100), np.linspace(0,25,num=100), 'k--', linewidth=0.5)
                ax2[1].set_xlabel('point count sorting [mm]')
                ax2[1].set_ylabel('DGS sorting [mm]')
                ax2[1].set_xlim([0, 13])
                ax2[1].set_ylim([0, 25])
                ax2[1].autoscale(enable=True, axis='x', tight=True)
                ax2[1].autoscale(enable=True, axis='y', tight=True)

        #         if n is 0:
        #             l1, = ax2[1,0].plot(mgs_g_ptc, mgs_g, 'k.', label='no window')
        #         else:
        #             l2, = ax2[1,0].plot(mgs_g_ptc, mgs_g, 'r.', label='window')
        #         # plt.plot(mgs_a_sieve, dgsjnk['mean grain size'], 'k.')
        # #            plt.plot(mgs_a_sieve, perc_freqs, 'c.')
        #         # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
        #         ax2[1,0].plot(np.arange(30), np.arange(30))
        #         ax2[1,0].set_ylabel('DGS geom. mgs [mm]')
        #         ax2[1,0].set_xlabel('pt count geom. mgs [mm]')
        #         if n is 0:
        #             # ax1[1,0].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'k.')
        #             ax2[1,1].plot(sort_g_ptc, sort_g, 'k.')
        #         else:
        #             # ax1[1,0].plot(sort_a_sieve, np.mean(dgsjnk['grain size sorting']), 'r.')
        #             ax2[1,1].plot(sort_g_ptc, sort_g, 'r.')
        #         ax2[1,1].plot(np.arange(20), np.arange(20))
        #         ax2[1,1].set_xlabel('pt count geom. sort [mm]')
        #         ax2[1,1].set_ylabel('DGS geom. sort [mm]')

                fig2.tight_layout()


            if os.path.exists(os.path.join(pointcountdir,  date_str0[ii] + tide0[ii] + '_' + depths[Igarbo] + '.npy')):
                if n is 0:
                    l31, = ax3[0,0].plot(mgs_a_ptc, mgs_a_sieve, 'k.', label='no window')
                else:
                    l32, = ax3[0,0].plot(mgs_a_ptc, mgs_a_sieve, 'r.', label='window')
                # plt.plot(mgs_a_sieve, dgsjnk['mean grain size'], 'k.')
        #            plt.plot(mgs_a_sieve, perc_freqs, 'c.')
                # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
                ax3[0,0].plot(np.arange(30), np.arange(30))
                ax3[0,0].set_ylabel('DGS arith. mgs [mm]')
                ax3[0,0].set_xlabel('pt count arith. mgs [mm]')
                if n is 0:
                    # ax1[1,0].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'k.')
                    ax3[0,1].plot(sort_a_ptc, sort_a_sieve, 'k.')
                else:
                    # ax1[1,0].plot(sort_a_sieve, np.mean(dgsjnk['grain size sorting']), 'r.')
                    ax3[0,1].plot(sort_a_ptc, sort_a_sieve, 'r.')
                ax3[0,1].plot(np.arange(20), np.arange(20))
                ax3[0,1].set_xlabel('pt count arith. sort [mm]')
                ax3[0,1].set_ylabel('sieve arith. sort [mm]')

                if n is 0:
                    l31, = ax3[1,0].plot(mgs_g_ptc, mgs_g_sieve, 'k.', label='no window')
                else:
                    l32, = ax3[1,0].plot(mgs_g_ptc, mgs_g_sieve, 'r.', label='window')
                # plt.plot(mgs_a_sieve, dgsjnk['mean grain size'], 'k.')
        #            plt.plot(mgs_a_sieve, perc_freqs, 'c.')
                # ax1[0].title('mean. RMSE = ' + str(rmse_mgs))
                ax3[1,0].plot(np.arange(30), np.arange(30))
                ax3[1,0].set_ylabel('DGS geom. mgs [mm]')
                ax3[1,0].set_xlabel('pt count geom. mgs [mm]')
                if n is 0:
                    # ax1[1,0].plot(sort_a_sieve, dgsjnk['grain size sorting'], 'k.')
                    ax3[1,1].plot(sort_g_ptc, sort_g_sieve, 'k.')
                else:
                    # ax1[1,0].plot(sort_a_sieve, np.mean(dgsjnk['grain size sorting']), 'r.')
                    ax3[1,1].plot(sort_g_ptc, sort_g_sieve, 'r.')
                ax3[1,1].plot(np.arange(20), np.arange(20))
                ax3[1,1].set_xlabel('pt count geom. sort [mm]')
                ax3[1,1].set_ylabel('sieve geom. sort [mm]')

                fig3.tight_layout()






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
    #        plt.plot(gsbins, gsfreq_sieve)
    #        plt.plot(mgs_a_sieve, 0, 'ro')

    #        plt.figure(9999)
    #        plt.plot(gsbins,  bar['cumulative sum'])
    #        plt.plot(xp, cumsum_interp,'r')

    #        plt.figure(1000+ii)
    #        plt.plot(jnk['percentile_values'], jnk['percentiles'])
    #        plt.plot(gsbins,  bar['cumulative sum'])
    #        plt.xlabel('values')
    #        plt.ylabel('percentiles')

    #        plt.plot(gsbins, gsfreq_sieve)
    #        plt.plot(mgs_a_sieve, 0, 'ro')

    #    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,7))
    #    pos1 = ax1.imshow(gsize['mean_grain_size'], cmap='inferno')
    #    fig.colorbar(pos1, ax=ax1)
    #    pos2 = ax2.imshow(gsize['sorting'], cmap='inferno')
    #    fig.colorbar(pos2, ax=ax2)

    rmse_mgs_sieve_dgs.append(np.sqrt(np.nanmean(np.array(rms_mgs_vec_dgs_sieve)**2)))
    rmse_mgs_count_dgs.append(np.sqrt(np.nanmean(np.array(rms_mgs_vec_dgs_count)**2)))
    rmse_sort = np.sqrt(np.nanmean(np.array(rms_std_vec)**2))

# ax1[0].legend(['no windowing', 'windowing'], loc="upper left")
# ax1[0,0].legend([l1, l2],['x=1.0', 'x=1.5'], loc="upper left")
ax1[0].legend([l0, l1, l2],['x=0.5', 'x=0.8', 'x=1.5'], loc="upper left")
ax2[0].legend([l00, l01, l02],['x=0.5', 'x=0.8', 'x=1.5'], loc="upper left")


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
fig4.tight_layout()


fig5, ax5 = plt.subplots(nrows=1, ncols=1, num='dist')
ax5.plot(dgsjnk['grain size bins'], dgsjnk['grain size frequencies'], 'k.', label='no window')
ax5.plot(bar['grain size bins'], bar['grain size frequencies'], 'C0', label='no window')
# ax5.set_ylabel('DGS mean [mm]')
fig5.tight_layout()

np.sum(bar['grain size frequencies'])


# EXPORT PLOTS
saveFlag = 0

if saveFlag == 1:

    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','dgs_validation')

    save_figures(savedn, 'DGS_sieve_validation', fig1)
    save_figures(savedn, 'DGS_ptcount_validation', fig2)

    # 2453*0.12/56
