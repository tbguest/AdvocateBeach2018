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

wave_height_threshold = 25 # set really high to omit
steepness_threshold = 0.01 # set really high to omit

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux

figsdn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures')

# cross-shore HWL coord and tide index
# hwl = [-21,-9,-15,-15,-18,-15,-15,-15,-18,-21,-18,-18,-18,-18]
hwl = [-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27,-27]
Ihwl = [14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26, 27]

# grid_specs = ["cross_shore"]#["longshore1", "longshore2"]#, "dense_array2"]
# grid_specs = ["cross_shore","longshore1", "dense_array2"]
# grid_specs = ["cross_shore","longshore1", "longshore2", "dense_array2"]
# grid_specs = ["longshore1", "longshore2", "dense_array2"]
# grid_specs = ["longshore1"]#, "longshore1", "longshore2", "dense_array2"]
grid_specs = ["longshore1"]

# guess at cross-shore profile length
npts = 150

# berm_zone_width = 6 # [m] should be multiple of 3
# hwl_adjust = 3
berm_zone_width = 20 # [m] should be multiple of 3
hwl_adjust = 0

above_HWL_dz = []
above_HWL_mgs = []
above_HWL_dmgs = []
above_HWL_last_mgs = []
above_HWL_last_dmgs = []
above_HWL_last_dz = []
above_HWL_dz_for_last_dz = []
above_HWL_dmgs_for_last_dmgs = []


at_HWL_dz = []
at_HWL_mgs = []
at_HWL_dmgs = []
at_HWL_last_mgs = []
at_HWL_last_dmgs = []
at_HWL_last_dz = []
at_HWL_dz_for_last_dz = []
at_HWL_dmgs_for_last_dmgs = []


below_HWL_dz = []
below_HWL_mgs = []
below_HWL_dmgs = []
below_HWL_last_mgs = []
below_HWL_last_dmgs = []
below_HWL_last_dz = []
below_HWL_dz_for_last_dz = []
below_HWL_dmgs_for_last_dmgs = []


# for tide comparison with wave data
below_HWL_dz_tides = {}
below_HWL_mgs_tides = {}
below_HWL_dmgs_tides = {}

at_HWL_dz_tides = {}
at_HWL_mgs_tides = {}
at_HWL_dmgs_tides = {}


accretion_bin_dz = []
accretion_bin_mgs = []
accretion_bin_dmgs = []
erosion_bin_dz = []
erosion_bin_mgs = []
erosion_bin_dmgs = []
nochange_bin_dz = []
nochange_bin_mgs = []
nochange_bin_dmgs = []

# # wave data
# Hs = []
# Tp = []
# steepness = []
# iribarren = []
# wave_energy = []
# wave_energy_wind = []
# wave_energy_swell = []
# maxdepth = []

Hs = {}
Tp = {}
steepness = {}
iribarren = {}
wave_energy = {}
wave_energy_wind = {}
wave_energy_swell = {}
maxdepth = {}

fig99, ax99 = plt.subplots(1,1, figsize=(4.2,3), num='regions')


for grid_spec in grid_specs:
    # grid_spec = "longshore2"
    # grid_spec = "longshore1"
    # grid_spec = "cross_shore"
    # grid_spec = "dense_array2"


    if grid_spec == 'cross_shore':
        start_tide = 13
    elif grid_spec == 'longshore1':
        start_tide = 15
        # hwl = hwl[2:]
    else:
        start_tide = 14
        # hwl = hwl[1:]

    tide_range = range(start_tide, 28)
    tide_axis = np.arange(start_tide+1,28) # for plotting later

    counter = 0

    # # initialize
    # all_dz = []
    # all_mgs = []
    # all_dmgs = []
    # all_last_mgs = []
    # all_last_dz = []


    for ii in tide_range:

        below_HWL_dz_tides_tmp = []
        below_HWL_mgs_tides_tmp = []
        below_HWL_dmgs_tides_tmp = []

        at_HWL_dz_tides_tmp = []
        at_HWL_mgs_tides_tmp = []
        at_HWL_dmgs_tides_tmp = []

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

        # if this hasn't already been done for different grid_spec
        if not ii in Hs:

            with open(wavefn, 'r') as fpp:
                bar = json.load(fpp)

            Hs[ii] = np.mean(np.array(bar["Hs"]))
            Tp[ii] = np.mean(np.array(bar["Tp"]))
            steepness[ii] = np.mean(np.array(bar["steepness"]))
            iribarren[ii] = np.mean(np.array(bar["Iribarren"]))
            wave_energy[ii] = np.mean(np.array(bar["wave_energy"]))
            wave_energy_wind[ii] = np.mean(np.array(bar["wave_energy_wind"]))
            wave_energy_swell[ii] = np.mean(np.array(bar["wave_energy_swell"]))
            maxdepth[ii] = np.max(bar["depth"])


        # sediment data
        # jnk = np.load(gsizefn, allow_pickle=True).item()
        with open(gsizefn, 'r') as fpp:
            jnk = json.load(fpp)

        mgs0 = np.array(jnk['mean_grain_size'])
        sort0 = np.array(jnk['sorting'])

        if grid_spec == 'dense_array2':
            mgs0 = np.squeeze(np.reshape(np.array(jnk['mean_grain_size']), (1,mgs0.shape[0]*mgs0.shape[1])))
            sort0 = np.squeeze(np.reshape(np.array(jnk['sorting']), (1,sort0.shape[0]*sort0.shape[1])))

        # np.reshape(mgs0, (1,mgs0.shape[0]*mgs0.shape[1]))

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
        yjunk = np.pad(y0, (0, npts-len(y0)), 'constant', constant_values=(np.nan,np.nan))
        y = np.around(yjunk)
        z = np.pad(z0, (0, npts-len(z0)), 'constant', constant_values=(np.nan,np.nan))

        # averaged wave data
        # Hs.append...


        # for first iteration - no data logging yet
        if counter == 0:

            last_z = np.copy(z)
            dz = z - last_z
            last_dz = np.copy(dz)

            last_mgs = np.copy(mgs)
            last_sort = np.copy(sort)
            dmgs = mgs - last_mgs
            dsort = sort - last_sort


        else:

            dz = z - last_z
            dmgs = mgs - last_mgs
            dsort = sort - last_sort


            if grid_spec == 'cross_shore':
                maxIy = 12 # deals with different length profiles
            else:
                maxIy = len(y)

            # populate bin based on cross-shore beach region (rel. to HWL)
            for yi in range(maxIy):

                if ~np.isnan(y[yi]):

                    if y[yi] < (hwl[Ihwl.index(ii)] - hwl_adjust):

                        ax99.plot(ii, y[yi], 'b.')

                        # wave height thresh
                        if Hs[ii] < wave_height_threshold:
                        # if steepness[ii] > steepness_threshold:

                            above_HWL_dz.append(dz[yi])
                            above_HWL_mgs.append(mgs[yi])
                            above_HWL_dmgs.append(dmgs[yi])
                            above_HWL_last_mgs.append(last_mgs[yi])
                            if counter > 1:
                                above_HWL_last_dz.append(last_dz[yi])
                                above_HWL_last_dmgs.append(last_dmgs[yi])
                                above_HWL_dz_for_last_dz.append(dz[yi]) # so lengths line up
                                above_HWL_dmgs_for_last_dmgs.append(dmgs[yi]) # so lengths line up

                    else:

                        # # for comparison with tide-averaged wave data PICK UP HERE...
                        # below_HWL_dz_tides_tmp.append(dz[yi])
                        # below_HWL_mgs_tides_tmp.append(mgs[yi])
                        # below_HWL_dmgs_tides_tmp.append(dmgs[yi])

                        if y[yi] - hwl[Ihwl.index(ii)] < berm_zone_width:

                            ax99.plot(ii, y[yi], 'g.')

                            # # for comparison with tide-averaged wave data PICK UP HERE...
                            # below_HWL_dz_tides_tmp.append(dz[yi])
                            # below_HWL_mgs_tides_tmp.append(mgs[yi])
                            # below_HWL_dmgs_tides_tmp.append(dmgs[yi])

                            # wave height thresh
                            if Hs[ii] < wave_height_threshold:
                            # if steepness[ii] > steepness_threshold:

                                # if np.abs(dz[yi]) > 0.02:

                                at_HWL_dz.append(dz[yi])
                                at_HWL_mgs.append(mgs[yi])
                                at_HWL_dmgs.append(dmgs[yi])
                                at_HWL_last_mgs.append(last_mgs[yi])

                                at_HWL_dz_tides_tmp.append(dz[yi])
                                at_HWL_mgs_tides_tmp.append(mgs[yi])
                                at_HWL_dmgs_tides_tmp.append(dmgs[yi])

                                if counter > 1:
                                    at_HWL_last_dz.append(last_dz[yi])
                                    at_HWL_last_dmgs.append(last_dmgs[yi])
                                    at_HWL_dz_for_last_dz.append(dz[yi])
                                    at_HWL_dmgs_for_last_dmgs.append(dmgs[yi])

                        else:

                            ax99.plot(ii, y[yi], 'r.')

                            # wave height thresh
                            if Hs[ii] < wave_height_threshold:
                            # if steepness[ii] > steepness_threshold:

                                # if np.abs(dz[yi]) > 0.02:

                                below_HWL_dz.append(dz[yi])
                                below_HWL_mgs.append(mgs[yi])
                                below_HWL_dmgs.append(dmgs[yi])
                                below_HWL_last_mgs.append(last_mgs[yi])

                                # for comparison with tide-averaged wave data PICK UP HERE...
                                below_HWL_dz_tides_tmp.append(dz[yi])
                                below_HWL_mgs_tides_tmp.append(mgs[yi])
                                below_HWL_dmgs_tides_tmp.append(dmgs[yi])


                                if counter > 1:
                                    below_HWL_last_dz.append(last_dz[yi])
                                    below_HWL_last_dmgs.append(last_dmgs[yi])
                                    below_HWL_dz_for_last_dz.append(dz[yi])
                                    below_HWL_dmgs_for_last_dmgs.append(dmgs[yi])

                            # for Masselink et al 2007 analysis:
                            if dz[yi] > 0.02:
                                accretion_bin_dz.append(dz[yi])
                                accretion_bin_mgs.append(mgs[yi])
                                accretion_bin_dmgs.append(dmgs[yi])
                            elif dz[yi] < -0.02:
                                erosion_bin_dz.append(dz[yi])
                                erosion_bin_mgs.append(mgs[yi])
                                erosion_bin_dmgs.append(dmgs[yi])
                            else:
                                nochange_bin_dz.append(dz[yi])
                                nochange_bin_mgs.append(mgs[yi])
                                nochange_bin_dmgs.append(dmgs[yi])


            # for comparison with tide-averaged wave data PICK UP HERE...
            if ii not in below_HWL_dz_tides:
                below_HWL_dz_tides[ii] = below_HWL_dz_tides_tmp
                below_HWL_mgs_tides[ii] = below_HWL_mgs_tides_tmp
                below_HWL_dmgs_tides[ii] = below_HWL_dmgs_tides_tmp
            else:
                below_HWL_dz_tides[ii].extend(below_HWL_dz_tides_tmp)
                below_HWL_mgs_tides[ii].extend(below_HWL_mgs_tides_tmp)
                below_HWL_dmgs_tides[ii].extend(below_HWL_dmgs_tides_tmp)

            if ii not in at_HWL_dz_tides:
                at_HWL_dz_tides[ii] = at_HWL_dz_tides_tmp
                at_HWL_mgs_tides[ii] = at_HWL_mgs_tides_tmp
                at_HWL_dmgs_tides[ii] = at_HWL_dmgs_tides_tmp
            else:
                at_HWL_dz_tides[ii].extend(at_HWL_dz_tides_tmp)
                at_HWL_mgs_tides[ii].extend(at_HWL_mgs_tides_tmp)
                at_HWL_dmgs_tides[ii].extend(at_HWL_dmgs_tides_tmp)


            # update
            last_z = np.copy(z)
            last_dz = np.copy(dz)
            last_mgs = np.copy(mgs)
            last_dmgs = np.copy(dmgs)
            last_sort = np.copy(sort)



        counter = counter + 1


# this is for removing the mean from each chunk, though im not sure it;s the right thing to do... UNFINISHED
below_HWL_dz_detrend = []
below_HWL_dmgs_detrend = []
below_HWL_mgs_detrend = []
at_HWL_dz_detrend = []
at_HWL_dmgs_detrend = []
at_HWL_mgs_detrend = []

for jj in range(tide_range[0]+1,tide_range[-1]):
    below_HWL_dz_detrend.extend(below_HWL_dz_tides[jj] - np.nanmean(below_HWL_dz_tides[jj]))
    below_HWL_dmgs_detrend.extend(below_HWL_dmgs_tides[jj] - np.nanmean(below_HWL_dmgs_tides[jj]))
    below_HWL_mgs_detrend.extend(below_HWL_mgs_tides[jj])
    at_HWL_dz_detrend.extend(at_HWL_dz_tides[jj] - np.nanmean(at_HWL_dz_tides[jj]))
    at_HWL_dmgs_detrend.extend(at_HWL_dmgs_tides[jj] - np.nanmean(at_HWL_dmgs_tides[jj]))
    at_HWL_mgs_detrend.extend(at_HWL_mgs_tides[jj])



# len(below_HWL_dz_tides[16])
# # 2, 26, 125, , , , ... 114
# # 6, 2, 29 ..., 70
# # 8, 28, 154... , 184

# wave data
Hs_array = []
Tp_array = []
steepness_array = []
iribarren_array = []
wave_energy_array = []
wave_energy_wind_array = []
wave_energy_swell_array = []
maxdepth_array = []

r_mgs_tides = np.zeros(len(below_HWL_dz_tides))
p_mgs_tides = np.zeros(len(below_HWL_dz_tides))
lo_mgs_tides = np.zeros(len(below_HWL_dz_tides))
hi_mgs_tides = np.zeros(len(below_HWL_dz_tides))
r_dmgs_tides = np.zeros(len(below_HWL_dz_tides))
p_dmgs_tides = np.zeros(len(below_HWL_dz_tides))
lo_dmgs_tides = np.zeros(len(below_HWL_dz_tides))
hi_dmgs_tides = np.zeros(len(below_HWL_dz_tides))
mean_mgs_tides = np.zeros(len(below_HWL_dz_tides))
std_mgs_tides = np.zeros(len(below_HWL_dz_tides))
mean_dmgs_tides = np.zeros(len(below_HWL_dz_tides))
mean_dz_tides = np.zeros(len(below_HWL_dz_tides))



# consolidate wave data, and do tide-dependent analysis:
count = -1
for jj in tide_range:
    count += 1

    Hs_array.append(Hs[jj])
    Tp_array.append(Tp[jj])
    steepness_array.append(steepness[jj])
    iribarren_array.append(iribarren[jj])
    wave_energy_array.append(wave_energy[jj])
    wave_energy_wind_array.append(wave_energy_wind[jj])
    wave_energy_swell_array.append(wave_energy_swell[jj])
    maxdepth_array.append(maxdepth[jj])

    # mean_dz_tides[count] = np.mean(below_HWL_dz_tides[jj])
    # mean_mgs_tides[count] = np.mean(below_HWL_mgs_tides[jj])
    # std_mgs_tides[count] = np.std(below_HWL_mgs_tides[jj])
    # mean_dmgs_tides[count] = np.mean(below_HWL_dmgs_tides[jj])


    # r_mgs_tides[count], p_mgs_tides[count], lo_mgs_tides[count], hi_mgs_tides[count] = pearsonr_ci(below_HWL_dz_tides[jj], below_HWL_mgs_tides[jj],alpha=0.05)
    # r_dmgs_tides[count], p_dmgs_tides[count], lo_dmgs_tides[count], hi_dmgs_tides[count] = pearsonr_ci(below_HWL_dz_tides[jj], below_HWL_dmgs_tides[jj],alpha=0.05)




# I_p_small_mgs = np.where(p_mgs_tides > 0.05)
I_p_small_mgs = np.where(r_mgs_tides < 0)
r_mgs_tides_signif = np.copy(r_mgs_tides)
r_mgs_tides_signif[I_p_small_mgs] = np.nan

# I_p_small_dmgs = np.where(p_dmgs_tides > 0.05)
I_p_small_dmgs = np.where(r_dmgs_tides < 0)
r_dmgs_tides_signif = np.copy(r_dmgs_tides)
r_dmgs_tides_signif[I_p_small_dmgs] = np.nan


# pearsonr, alphar = pearsonr(below_HWL_dz, below_HWL_last_mgs)

## TESTING
# test_data_dmgs = []
# test_data_dz = []
#
# for jj in tide_range:
#     test_data_dmgs.extend(below_HWL_dmgs_tides[jj])
#     test_data_dz.extend(below_HWL_dz_tides[jj])
#
# # test
# r_dmgs_test, p_dmgs_test, lo_dmgs_test, hi_dmgs_test = pearsonr_ci(test_data_dz,test_data_dmgs,alpha=0.05)
# r_dmgs, p_dmgs, lo_dmgs, hi_dmgs = pearsonr_ci(below_HWL_dz,below_HWL_dmgs,alpha=0.05)
#
# len(test_data)
# len(below_HWL_dmgs)
#
# np.array(test_data) - np.array(below_HWL_dmgs)

# len(below_HWL_dz)
# len(at_HWL_dz)
# len(above_HWL_dz)

# r_mgs, p_mgs, lo_mgs, hi_mgs = pearsonr_ci(below_HWL_dz_detrend,below_HWL_mgs_detrend,alpha=0.05)
# r_dmgs, p_dmgs, lo_dmgs, hi_dmgs = pearsonr_ci(below_HWL_dz_detrend,below_HWL_dmgs_detrend,alpha=0.05)
# r_last_mgs, p_last_mgs, lo_last_mgs, hi_last_mgs = pearsonr_ci(below_HWL_dz,below_HWL_last_mgs,alpha=0.05)
# r_last_dz, p_last_dz, lo_last_dz, hi_last_dz = pearsonr_ci(below_HWL_dz_for_last_dz,below_HWL_last_dz,alpha=0.05)
r_mgs, p_mgs, lo_mgs, hi_mgs = pearsonr_ci(below_HWL_dz,below_HWL_mgs,alpha=0.05)
r_dmgs, p_dmgs, lo_dmgs, hi_dmgs = pearsonr_ci(below_HWL_dz,below_HWL_dmgs,alpha=0.05)
r_last_mgs, p_last_mgs, lo_last_mgs, hi_last_mgs = pearsonr_ci(below_HWL_dz,below_HWL_last_mgs,alpha=0.05)
r_last_dz, p_last_dz, lo_last_dz, hi_last_dz = pearsonr_ci(below_HWL_dz_for_last_dz,below_HWL_last_dz,alpha=0.05)
r_mgs_last_mgs, p_mgs_last_mgs, lo_mgs_last_mgs, hi_mgs_last_mgs = pearsonr_ci(below_HWL_mgs,below_HWL_last_mgs,alpha=0.05)
r_dmgs_last_dmgs, p_dmgs_last_dmgs, lo_dmgs_last_dmgs, hi_dmgs_last_dmgs = pearsonr_ci(below_HWL_dmgs_for_last_dmgs,below_HWL_last_dmgs,alpha=0.05)


# at_r_mgs, at_p_mgs, at_lo_mgs, at_hi_mgs = pearsonr_ci(at_HWL_dz_detrend,at_HWL_mgs_detrend,alpha=0.05)
# at_r_dmgs, at_p_dmgs, at_lo_dmgs, at_hi_dmgs = pearsonr_ci(at_HWL_dz_detrend,at_HWL_dmgs_detrend,alpha=0.05)
# at_r_last_mgs, at_p_last_mgs, at_lo_last_mgs, at_hi_last_mgs = pearsonr_ci(at_HWL_dz,at_HWL_last_mgs,alpha=0.05)
# at_r_last_dz, at_p_last_dz, at_lo_last_dz, at_hi_last_dz = pearsonr_ci(at_HWL_dz_for_last_dz,at_HWL_last_dz,alpha=0.05)
at_r_mgs, at_p_mgs, at_lo_mgs, at_hi_mgs = pearsonr_ci(at_HWL_dz,at_HWL_mgs,alpha=0.05)
at_r_dmgs, at_p_dmgs, at_lo_dmgs, at_hi_dmgs = pearsonr_ci(at_HWL_dz,at_HWL_dmgs,alpha=0.05)
at_r_last_mgs, at_p_last_mgs, at_lo_last_mgs, at_hi_last_mgs = pearsonr_ci(at_HWL_dz,at_HWL_last_mgs,alpha=0.05)
at_r_last_dz, at_p_last_dz, at_lo_last_dz, at_hi_last_dz = pearsonr_ci(at_HWL_dz_for_last_dz,at_HWL_last_dz,alpha=0.05)
at_r_mgs_last_mgs, at_p_mgs_last_mgs, at_lo_mgs_last_mgs, at_hi_mgs_last_mgs = pearsonr_ci(at_HWL_mgs,at_HWL_last_mgs,alpha=0.05)
at_r_dmgs_last_dmgs, at_p_dmgs_last_dmgs, at_lo_dmgs_last_dmgs, at_hi_dmgs_last_dmgs = pearsonr_ci(at_HWL_dmgs_for_last_dmgs,at_HWL_last_dmgs,alpha=0.05)


above_r_mgs, above_p_mgs, above_lo_mgs, above_hi_mgs = pearsonr_ci(above_HWL_dz,above_HWL_mgs,alpha=0.05)
above_r_dmgs, above_p_dmgs, above_lo_dmgs, above_hi_dmgs = pearsonr_ci(above_HWL_dz,above_HWL_dmgs,alpha=0.05)
above_r_last_mgs, above_p_last_mgs, above_lo_last_mgs, above_hi_last_mgs = pearsonr_ci(above_HWL_dz,above_HWL_last_mgs,alpha=0.05)
above_r_last_dz, above_p_last_dz, above_lo_last_dz, above_hi_last_dz = pearsonr_ci(above_HWL_dz_for_last_dz,above_HWL_last_dz,alpha=0.05)
above_r_mgs_last_mgs, above_p_mgs_last_mgs, above_lo_mgs_last_mgs, above_hi_mgs_last_mgs = pearsonr_ci(above_HWL_mgs,above_HWL_last_mgs,alpha=0.05)
above_r_dmgs_last_dmgs, above_p_dmgs_last_dmgs, above_lo_dmgs_last_dmgs, above_hi_dmgs_last_dmgs = pearsonr_ci(above_HWL_dmgs_for_last_dmgs,above_HWL_last_dmgs,alpha=0.05)

# r_mgs, p_mgs, lo_mgs, hi_mgs
# r_dmgs, p_dmgs, lo_dmgs, hi_dmgs
# r_last_mgs, p_last_mgs, lo_last_mgs, hi_last_mgs
# r_last_dz, p_last_dz, lo_last_dz, hi_last_dz
#
# at_r_mgs, at_p_mgs, at_lo_mgs, at_hi_mgs
# at_r_dmgs, at_p_dmgs, at_lo_dmgs, at_hi_dmgs
# at_r_last_mgs, at_p_last_mgs, at_lo_last_mgs, at_hi_last_mgs
# at_r_last_dz, at_p_last_dz, at_lo_last_dz, at_hi_last_dz
#
# above_r_mgs, above_p_mgs, above_lo_mgs, above_hi_mgs
# above_r_dmgs, above_p_dmgs, above_lo_dmgs, above_hi_dmgs
# above_r_last_mgs, above_p_last_mgs, above_lo_last_mgs, above_hi_last_mgs
# above_r_last_dz, above_p_last_dz, above_lo_last_dz, above_hi_last_dz


# Masselink et al 2007 analysis:
r_accr_mgs, p_accr_mgs, lo_accr_mgs, hi_accr_mgs = pearsonr_ci(accretion_bin_dz,accretion_bin_mgs,alpha=0.05)
r_accr_dmgs, p_accr_dmgs, lo_accr_dmgs, hi_accr_dmgs = pearsonr_ci(accretion_bin_dz,accretion_bin_dmgs,alpha=0.05)
r_eros_mgs, p_eros_mgs, lo_eros_mgs, hi_eros_mgs = pearsonr_ci(erosion_bin_dz,erosion_bin_mgs,alpha=0.05)
r_eros_dmgs, p_eros_dmgs, lo_eros_dmgs, hi_eros_dmgs = pearsonr_ci(erosion_bin_dz,erosion_bin_dmgs,alpha=0.05)


# accretion
np.mean(accretion_bin_mgs)
np.std(accretion_bin_mgs)

# no change
np.mean(nochange_bin_mgs)
np.std(nochange_bin_mgs)

# erosion
np.mean(erosion_bin_mgs)
np.std(erosion_bin_mgs)


msg_positive_change = np.squeeze(np.where(np.array(accretion_bin_dmgs) > 0))
msg_positive_change_pct = len(msg_positive_change)/len(accretion_bin_dmgs)

msg_negative_change = np.squeeze(np.where(np.array(erosion_bin_dmgs) < 0))
msg_negative_change_pct = len(msg_negative_change)/len(erosion_bin_dmgs)


len(accretion_bin_mgs)
len(nochange_bin_mgs)
len(erosion_bin_mgs)

len(msg_positive_change) - len(accretion_bin_dmgs)
len(msg_negative_change) - len(erosion_bin_dmgs)




# plot
fig, ax = plt.subplots(3,4, figsize=(10,5), num='scatter plots')

ax[0,0].plot(above_HWL_dz,above_HWL_mgs, '.')
ax[0,1].plot(above_HWL_dz,above_HWL_dmgs, '.')
ax[0,2].plot(above_HWL_dz,above_HWL_last_mgs, '.')
ax[0,3].plot(above_HWL_dz_for_last_dz,above_HWL_last_dz, '.')

# ax[1,0].plot(at_HWL_dz,at_HWL_mgs, '.')
# ax[1,1].plot(at_HWL_dz,at_HWL_dmgs, '.')
ax[1,0].plot(at_HWL_dz_detrend,at_HWL_mgs_detrend, '.')
ax[1,1].plot(at_HWL_dz_detrend,at_HWL_dmgs_detrend, '.')
ax[1,2].plot(at_HWL_dz,at_HWL_last_mgs, '.')
ax[1,3].plot(at_HWL_dz_for_last_dz,at_HWL_last_dz, '.')

# ax[2,0].plot(below_HWL_dz,below_HWL_mgs, '.')
# ax[2,1].plot(below_HWL_dz,below_HWL_dmgs, '.')
ax[2,0].plot(below_HWL_dz_detrend,below_HWL_mgs_detrend, '.')
ax[2,1].plot(below_HWL_dz_detrend,below_HWL_dmgs_detrend, '.')
ax[2,2].plot(below_HWL_dz,below_HWL_last_mgs, '.')
ax[2,3].plot(below_HWL_dz_for_last_dz,below_HWL_last_dz, '.')

fig.tight_layout()


# # plot
# fig, ax = plt.subplots(3,4, figsize=(10,5), num='scatter plots - detrended')
#
# ax[0,0].plot(above_HWL_dz,above_HWL_mgs, '.')
# ax[0,1].plot(above_HWL_dz,above_HWL_dmgs, '.')
# ax[0,2].plot(above_HWL_dz,above_HWL_last_mgs, '.')
# ax[0,3].plot(above_HWL_dz_for_last_dz,above_HWL_last_dz, '.')
#
# # ax[1,0].plot(at_HWL_dz,at_HWL_mgs, '.')
# # ax[1,1].plot(at_HWL_dz,at_HWL_dmgs, '.')
# ax[1,0].plot(at_HWL_dz_detrend,at_HWL_mgs_detrend, '.')
# ax[1,1].plot(at_HWL_dz_detrend,at_HWL_dmgs_detrend, '.')
# ax[1,2].plot(at_HWL_dz,at_HWL_last_mgs, '.')
# ax[1,3].plot(at_HWL_dz_for_last_dz,at_HWL_last_dz, '.')
#
# # ax[2,0].plot(below_HWL_dz,below_HWL_mgs, '.')
# # ax[2,1].plot(below_HWL_dz,below_HWL_dmgs, '.')
# ax[2,0].plot(below_HWL_dz_detrend,below_HWL_mgs_detrend, '.')
# ax[2,1].plot(below_HWL_dz_detrend,below_HWL_dmgs_detrend, '.')
# ax[2,2].plot(below_HWL_dz,below_HWL_last_mgs, '.')
# ax[2,3].plot(below_HWL_dz_for_last_dz,below_HWL_last_dz, '.')
#
# fig.tight_layout()



fig2, ax2 = plt.subplots(1,1, figsize=(4.2,3), num='corrs')

ax2.plot(above_r_mgs, 0.9, '.k')
ax2.plot(at_r_mgs, 1.0, '.r')
ax2.plot(r_mgs, 1.1, '.b')
ax2.plot([above_lo_mgs, above_hi_mgs], [0.9, 0.9], '-k')
ax2.plot([at_lo_mgs, at_hi_mgs], [1.0, 1.0], '-r')
ax2.plot([lo_mgs, hi_mgs], [1.1, 1.1], '-b')

ax2.plot(above_r_dmgs, 1.9, '.k')
ax2.plot([above_lo_dmgs, above_hi_dmgs], [1.9, 1.9], '-k')
ax2.plot(at_r_dmgs, 2.0, '.r')
ax2.plot([at_lo_dmgs, at_hi_dmgs], [2.0, 2.0], '-r')
ax2.plot(r_dmgs, 2.1, '.b')
ax2.plot([lo_dmgs, hi_dmgs], [2.1, 2.1], '-b')

ax2.plot(above_r_last_mgs, 2.9, '.k')
ax2.plot([above_lo_last_mgs, above_hi_last_mgs], [2.9, 2.9], '-k')
ax2.plot(at_r_last_mgs, 3.0, '.r')
ax2.plot([at_lo_last_mgs, at_hi_last_mgs], [3.0, 3.0], '-r')
ax2.plot(r_last_mgs, 3.1, '.b')
ax2.plot([lo_last_mgs, hi_last_mgs], [3.1, 3.1], '-b')

ax2.plot(above_r_last_dz, 3.9, '.k')
ax2.plot([above_lo_last_dz, above_hi_last_dz], [3.9, 3.9], '-k')
ax2.plot(at_r_last_dz, 4.0, '.r')
ax2.plot([at_lo_last_dz, at_hi_last_dz], [4.0, 4.0], '-r')
ax2.plot(r_last_dz, 4.1, '.b')
ax2.plot([lo_last_dz, hi_last_dz], [4.1, 4.1], '-b')

# ax2.plot(above_r_mgs_last_mgs, 4.9, '.k')
# ax2.plot([above_lo_mgs_last_mgs, above_hi_mgs_last_mgs], [4.9, 4.9], '-k')
# ax2.plot(at_r_mgs_last_mgs, 5.0, '.r')
# ax2.plot([at_lo_mgs_last_mgs, at_hi_mgs_last_mgs], [5.0, 5.0], '-r')
# ax2.plot(r_mgs_last_mgs, 5.1, '.b')
# ax2.plot([lo_mgs_last_mgs, hi_mgs_last_mgs], [5.1, 5.1], '-b')
#
# ax2.plot(above_r_dmgs_last_dmgs, 5.9, '.k')
# ax2.plot([above_lo_dmgs_last_dmgs, above_hi_dmgs_last_dmgs], [5.9, 5.9], '-k')
# ax2.plot(at_r_dmgs_last_dmgs, 6.0, '.r')
# ax2.plot([at_lo_dmgs_last_dmgs, at_hi_dmgs_last_dmgs], [6.0, 6.0], '-r')
# ax2.plot(r_dmgs_last_dmgs, 6.1, '.b')
# ax2.plot([lo_dmgs_last_dmgs, hi_dmgs_last_dmgs], [6.1, 6.1], '-b')


ax2.invert_yaxis()
ax2.plot([0,0], [0.8, 4.2], '--k')
ax2.autoscale(enable=True, axis='y', tight=True)
ax2.set_xlabel('$r$')

setstr = [r'$\Delta z$, MGS', r'$\Delta z$, $\Delta$MGS', r'$\Delta z$, MGS$_{t-1}$', r'$\Delta z$, $\Delta z_{t-1}$']#, r'MGS, MGS$_{t-1}$', r'dMGS, dMGS$_{t-1}$']
setind = [1,2,3,4]#, 5,6]
ax2.set_yticks(setind)
ax2.set_yticklabels(setstr)
ax2.legend(['above HWL,' + ' n=' + str(len(above_HWL_dz)), 'HWL,' + ' n=' + str(len(at_HWL_dz)), 'below HWL,' + ' n=' + str(len(below_HWL_dz))])

fig2.tight_layout()



# # tide dependent
# fig3, ax3 = plt.subplots(4,1, figsize=(4.2,9), num='tides')
#
# ax3[0].plot(mean_mgs_tides, Hs_array, '.')
# # ax3[0].plot(mean_dmgs_tides, Hs_array, '.')
# ax3[1].plot(mean_dz_tides, Hs_array, '.')
#
# ax3[2].plot(r_mgs_tides_signif[1:], Hs_array[:-1], '.')
# ax3[3].plot(r_dmgs_tides_signif[1:], Hs_array[:-1], '.')
#
# fig3.tight_layout()
#
#
# fig4, ax4 = plt.subplots(5,1, figsize=(4.2,10), num='tides 2')
#
# ax4[0].plot(p_mgs_tides, Hs_array, '.')
# ax4[0].plot(p_dmgs_tides, Hs_array, '.')
#
# ax4[1].plot(p_mgs_tides, Tp_array, '.')
# ax4[1].plot(p_dmgs_tides, Tp_array, '.')
#
# ax4[2].plot(p_mgs_tides, steepness_array, '.')
# ax4[2].plot(p_dmgs_tides, steepness_array, '.')
#
# ax4[3].plot(p_mgs_tides, iribarren_array, '.')
# ax4[3].plot(p_dmgs_tides, iribarren_array, '.')
#
# ax4[4].plot(p_mgs_tides, wave_energy_wind_array, '.')
# ax4[4].plot(p_dmgs_tides, wave_energy_wind_array, '.')
#
# fig4.tight_layout()
#
#
# fig5, ax5 = plt.subplots(4,1, figsize=(4.2,10), num='tides 3')
#
# ax5[0].plot(tide_range, Hs_array, '.')
# ax5[1].plot(tide_range, r_mgs_tides, '.')
#
# ax5[2].plot(tide_range, r_dmgs_tides, '.')
# ax5[3].plot(tide_range, p_dmgs_tides, '.')


# fig6, ax6 = plt.subplots(2,1, figsize=(4.2,6), num='Hsig and mgs')
#
# ax6[0].plot(tide_range, Hs_array, 'k.')
# ax6[1].plot(tide_range, mean_mgs_tides, 'k.')
# ax6[1].plot([tide_range, tide_range] , [mean_mgs_tides - std_mgs_tides, mean_mgs_tides + std_mgs_tides], 'k-')


saveFlag = 0
# EXPORT PLOTS
if saveFlag == 1:

    savedn = os.path.join(figsdn,'beach_profile')

    # save_figures(savedn, 'pearson_correlation_coefficients', fig)
    save_figures(savedn, 'all_survey_correlation_coefficients', fig2)



# ax2.plot(above_r_mgs,0.9, '.C0')
# ax2.plot([above_lo_mgs, above_hi_mgs], [0.9, 0.9], '-C0')
# ax2.plot(above_r_dmgs, 0.975, '.C1')
# ax2.plot([above_lo_dmgs, above_hi_dmgs],[0.975, 0.975], '-C1')
# ax2.plot(above_r_last_mgs, 1.025, '.C2')
# ax2.plot([above_lo_last_mgs, above_hi_last_mgs],[1.025, 1.025], '-C2')
# ax2.plot(above_r_last_dz, 1.1, '.C3')
# ax2.plot([above_lo_last_dz, above_hi_last_dz],[1.1, 1.1], '-C3')
#
# ax2.plot(at_r_mgs, 1.9, '.C0')
# ax2.plot([at_lo_mgs, at_hi_mgs], [1.9, 1.9], '-C0')
# ax2.plot(at_r_dmgs, 1.975, '.C1')
# ax2.plot([at_lo_dmgs, at_hi_dmgs], [1.975, 1.975], '-C1')
# ax2.plot(at_r_last_mgs, 2.025, '.C2')
# ax2.plot([at_lo_last_mgs, at_hi_last_mgs], [2.025, 2.025], '-C2')
# ax2.plot(at_r_last_dz, 2.1, '.C3')
# ax2.plot([at_lo_last_dz, at_hi_last_dz], [2.1, 2.1], '-C3')
#
# ax2.plot(r_mgs, 2.9, '.C0')
# ax2.plot([lo_mgs, hi_mgs], [2.9, 2.9], '-C0')
# ax2.plot(r_dmgs, 2.975, '.C1')
# ax2.plot([lo_dmgs, hi_dmgs], [2.975, 2.975], '-C1')
# ax2.plot(r_last_mgs, 3.025, '.C2')
# ax2.plot([lo_last_mgs, hi_last_mgs], [3.025, 3.025], '-C2')
# ax2.plot(r_last_dz, 3.1, '.C3')
# ax2.plot([lo_last_dz, hi_last_dz], [3.1, 3.1], '-C3')
#
# ax2.invert_yaxis()
# ax2.plot([0,0], [0.8, 3.2], '--k')
#
# fig.tight_layout()


    # # EXPORT PLOTS
    # if saveFlag == 1:
    #
    #     savedn = os.path.join(figsdn,'beach_profile',grid_spec)
    #     save_figures(savedn, 'pearson_correlation_coefficients', fig9)



    # if __name__ == '__main__':
    #     main()
7.2/2
