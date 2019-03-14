# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:00 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def main():
    
    # load in lookup table  

    # for portability
    homechar = "C:\\"      
    lookupdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "references", \
                          "img2sample_lookup_table.csv")

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
            

    
    date_str0 = ["Oct21", "Oct21", "Oct21", "Oct21", "Oct21", \
                 "Oct21", "Oct25", "Oct25", "Oct25", "Oct25"]
    tide0 = ["_bay1", "_bay2", "_bay3", "_horn1", "_horn2", "_horn3", \
             "_bay1", "_bay2", "_horn1", "_horn2"]   

    sample_loc = range(0, len(tide0))

    ## load key:value dict to convert from yearday to tide num.
    #tidekeydn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "external", "tide_key_values.npy")
    #tidekey = np.load(tidekeydn).item()
        
    rms_mgs_vec = []
    rms_std_vec = []
    
    for ii in sample_loc:
            
#        gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
#                          "grainsize_dists", "LabValidation_x0", date_str0[ii] + tide0[ii])
        
        gsizedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                          "grainsize", "validation", "LabValidation_xpos05", date_str0[ii] + tide0[ii])
        
#        C:\Projects\AdvocateBeach2018\data\processed\grainsize\validation\LabValidation_x0
        
        allfiles = os.listdir(gsizedir)
        
        # need to loop over files too
        
        for jj in range(0, len(allfiles)):
        
            jnk = np.load(os.path.join(gsizedir, allfiles[jj]), encoding='latin1').item()
            
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
            
            perc_freqs = jnk['percentile_values'][4]
            
            cumsum_img = np.cumsum(jnk['grain size frequencies'])
                        
            rms_mgs_vec.append(mean_gs - jnk['mean grain size'])
            rms_std_vec.append(std_gs - jnk['grain size sorting'])
            
            plt.figure(100)
            plt.subplot(221)
            plt.plot(mean_gs, jnk['mean grain size'], 'k.')
#            plt.plot(mean_gs, perc_freqs, 'c.')
#            plt.title('mean. RMSE = ' + str(rmse_mgs))
            plt.plot(np.arange(20), np.arange(20))
            plt.subplot(222)
            plt.plot(std_gs, jnk['grain size sorting'], 'k.')
            plt.plot(np.arange(20), np.arange(20))
#            plt.title('sorting. RMSE = ' + str(rmse_sort))
            plt.subplot(223)
            plt.plot(skew_gs, jnk['grain size skewness'], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('skewness')
            plt.xlabel('sieved')
            plt.ylabel('DGS')
            plt.subplot(224)
            plt.plot(kurt_gs, jnk['grain size kurtosis'], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('kurtosis')
            
            plt.figure(102)
            plt.subplot(231) 
            plt.plot(xp[p10], jnk['percentile_values'][1], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('10th percentile')
            plt.subplot(232)
            plt.plot(xp[p16], jnk['percentile_values'][2], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('16th percentile')
            plt.subplot(233)
            plt.plot(xp[p25], jnk['percentile_values'][3], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('25th percentile')
            plt.subplot(234)
            plt.plot(xp[p50], jnk['percentile_values'][4], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('50th percentile')
            plt.subplot(235)
            plt.plot(xp[p75], jnk['percentile_values'][5], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('75th percentile')
            plt.subplot(236)
            plt.plot(xp[p84], jnk['percentile_values'][6], 'k.')
            plt.plot(np.arange(20), np.arange(20))
            plt.title('84th percentile')
            
            plt.figure(103)
            plt.plot(xp[p50], jnk['percentile_values'][4], 'm.')
            plt.plot(np.arange(20), np.arange(20))
            plt.xlabel('d50')
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
    
#    print(rms_mgs_vec)
        
    plt.figure(100)
    plt.subplot(221)
    plt.title('mean. RMSE = ' + str(rmse_mgs))
    plt.subplot(222)
    plt.title('sorting. RMSE = ' + str(rmse_sort)) 
        
main()        
