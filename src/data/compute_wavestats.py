# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:38:23 2018

@author: Owner
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy import signal

%matplotlib qt5


def return_data(fname):

    jnk = np.load(fname)
    jnk = jnk[()]
    d = jnk['d']
    t = jnk['t']
    hightide = jnk['high_tide']

    return t, d, hightide


#def get_chunk(t, d):
#
#    dt = 12/(60*24) # 12 min averaging intervals
#
#    # define time intervals for averaging
#    nintervals = np.floor((np.max(t) - np.min(t))/dt)
#
#    # define indices of interval limits
#        mnind = np.argmin(np.abs(t - (t[0] + ii*dt)))
#        mxind = np.argmin(np.abs(t - (t[0] + (ii+1)*dt)))
#
#        # new truncated vectors
#        tt0 = t[mnind:mxind]
#        hh0 = d[mnind:mxind]
#        hmean = np.nanmean(hh0)
#
#        hh = hh0[~np.isnan(hh0)]
#        tt = tt0[~np.isnan(hh0)]


def implicit_wavenumber(F, hmean):

    ''' Takes Frequency vector returned by pwelch fuction, along with a mean
    water depth, to compute a wavenumber vector implicitly using the surface gravity
    wave dispersion relation.
    '''

    # valid for freq range...
    kspace = np.linspace(0,10,1000)

#    LHS = np.zeros(int(np.floor(len(F)/3)))
#    Ik = np.zeros(int(np.floor(len(F)/3)))
#    krad = np.zeros(int(np.floor(len(F)/3)))

    krad = [] # radian wavenumber

    for nn in range(0, int(len(F)/3)):

        LHS = (2.*np.pi*F[nn])**2
        RHS = g * np.multiply(kspace, np.tanh(kspace*hmean))

#        LHS[nn] = (2*np.pi*F[nn])**2
#        Ik[nn] = np.argmin(np.abs(LHS[nn] - RHS))
#        krad[nn] = kspace[int(Ik[nn])]

        Ik = np.argmin(np.abs(LHS - RHS))
        krad.append(kspace[Ik])

    return np.array(krad)



def atten_correct(Pxx, F, krad, hmean):

    # define pressure transfer function
    # Kp = cosh(krad*(hmean + (-hmean)))./cosh(krad*hmean), i.e.,
    Kp = 1/np.cosh(krad*hmean)
    Kp[85:] = 1 # nullify high frequency amplification

    # correct for depth attenuation
    Phh = Pxx[0:int(np.floor(len(F)/3))]/Kp

    return Phh


def compute_wavestats(Phh, F, hmean):

    # infragrav: 0.004:0.05 Hz (Lippmann et al 1999)
    # swell band: 0.05:0.15 Hz
    # wind band: 0.15:1 Hz (Masselink and Pattiaratchi JCR1998)
    I1_infr = np.argmin(np.abs(F - 0.004))
    I2_infr = np.argmin(np.abs(F - 0.05))
    I1_swel = np.argmin(np.abs(F - 0.05))
    I2_swel = np.argmin(np.abs(F - 0.15))
    I1_wind = np.argmin(np.abs(F - 0.15))
    # changed upper bound to 0.5 Hz to account for noise amplification
    I2_wind = np.argmin(np.abs(F - 0.5))
    # define general lower and upper boundaries (as defined by NOAA)
    I1 = np.argmin(np.abs(F - 0.0325))
    I2 = np.argmin(np.abs(F - 0.485))


    # m_0 (m_n is the nth moment of the spectrum of hh)
    m0_infr = np.sum(Phh[I1_infr:I2_infr])*np.mean(np.diff(F))
    m0_swel = np.sum(Phh[I1_swel:I2_swel])*np.mean(np.diff(F))
    m0_wind = np.sum(Phh[I1_wind:I2_wind])*np.mean(np.diff(F))
    m0 = np.sum(Phh[I1:I2])*np.mean(np.diff(F))

    # m_1
    m1_infr = np.sum(Phh[I1_infr:I2_infr]*F[I1_infr:I2_infr])*np.mean(np.diff(F))
    m1_swel = np.sum(Phh[I1_swel:I2_swel]*F[I1_swel:I2_swel])*np.mean(np.diff(F))
    m1_wind = np.sum(Phh[I1_wind:I2_wind]*F[I1_wind:I2_wind])*np.mean(np.diff(F))
    m1 = np.sum(Phh[I1:I2]*F[I1:I2])*np.mean(np.diff(F))

    # m_2
    m2_infr = np.sum(Phh[I1_infr:I2_infr]*F[I1_infr:I2_infr]**2)*np.mean(np.diff(F))
    m2_swel = np.sum(Phh[I1_swel:I2_swel]*F[I1_swel:I2_swel]**2)*np.mean(np.diff(F))
    m2_wind = np.sum(Phh[I1_wind:I2_wind]*F[I1_wind:I2_wind]**2)*np.mean(np.diff(F))
    m2 = np.sum(Phh[I1:I2]*F[I1:I2]**2)*np.mean(np.diff(F))

    # significant wave height
    Hs_infr = 4*np.sqrt(m0_infr)
    Hs_swell = 4*np.sqrt(m0_swel)
    Hs_wind = 4*np.sqrt(m0_wind)
    Hs = 4*np.sqrt(m0)

    # mean spectral period: Tmean = 2*pi*m_0/m_1 (omit 2*pi?)
    # mean spectral period: Tmean = sqrt(m_0/m_2)
    Tmean_infr = np.sqrt(m0_infr/m2_infr)
    Tmean_swell = np.sqrt(m0_swel/m2_swel)
    Tmean_wind = np.sqrt(m0_wind/m2_wind)
    Tmean = np.sqrt(m0/m2)

    # peak wave period
    offset = 3 # so energy near 0 Hz isn't counted
    peakind0 = np.argmax(Phh[offset:])
    peakind = peakind0 + offset
    Tp = 1/F[peakind]

    # linear dispersion relation for wavenumber computation
    kspace = np.linspace(0,10,1000)
    LHS = (2*np.pi*F[peakind])**2
    RHS = g*np.multiply(kspace, np.tanh(kspace*hmean))
    Ipk = np.argmin(np.abs(LHS - RHS))
    # in case of div by 0
    if Ipk == 0:
        Ipk = 1
    k_peak = kspace[Ipk]

    # "deep water" wavelength
    L = 2*np.pi/k_peak

    ###
#             *****why doing these with a separate timescale?
    # wave steepness
    steepness = Hs/L

    # Miche number
    M = 16*g**2*(np.tan(beta_slope)**5)/((2*np.pi)**5*Hs**2*F[peakind]**4)

    # Iribarren number
    xi_b = np.tan(beta_slope)/(steepness**0.5)

    # surf-scaling
    omeg = 2*np.pi/Tp
    a_b = Hs/2 # an approximation for wave runup
    eps_sc = a_b*omeg**2/(g*np.tan(beta_slope)**2)

    waves = {"Hs": Hs,
             "Hs_swell": Hs_swell,
             "Hs_wind": Hs_wind,
             "Tmean": Tmean,
             "Tmean_swell": Tmean_swell,
             "Tmean_wind": Tmean_wind,
             "Tp": Tp,
             "wavelength": L,
             "steepness": steepness,
             "Miche": M,
             "Iribarren": xi_b,
             "surf_scaling": eps_sc,}

    return waves





homechar = "C:\\"

dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                     "interim","pressure")

#fns = os.listdir(dn)

yd = range(9, 27)

beta_slope = 0.12

# sampling  frequency of RBR
fs = 4

rho_s = 1030 # [kg*m^-3] density of seawater
g = 9.81

# define time intervals for averaging
dt = 12/(60*24) # 12 min averaging intervals

# spectral analysis
nfft = 512
#nfft = ceil(fs/fResMin)
winhann = signal.hann(nfft, sym=False)
overlap = 0.5

for kk in yd:

    fn = "tide" + str(kk + 1) + ".npy"
    t, d, hightide = return_data(os.path.join(dn, fn))

    if kk == 26:
        plt.figure(99)
        plt.plot(t,d)

    # define time intervals for averaging
    nintervals = np.floor((np.max(t) - np.min(t))/dt)

    # initialize (new short time interval vectors)
    Hs_infr = np.zeros(int(nintervals))
    Hs_swell = np.zeros(int(nintervals))
    Hs_wind = np.zeros(int(nintervals))
    Hs = np.zeros(int(nintervals))
    Tp = np.zeros(int(nintervals))
    Tmean_infr = np.zeros(int(nintervals))
    Tmean_swell = np.zeros(int(nintervals))
    Tmean_wind = np.zeros(int(nintervals))
    Tmean = np.zeros(int(nintervals))
    L = np.zeros(int(nintervals))
    steepness = np.zeros(int(nintervals))
    M = np.zeros(int(nintervals))
    xi_b = np.zeros(int(nintervals))
    eps_sc = np.zeros(int(nintervals))
    depth = np.zeros(int(nintervals))
    timevec = np.zeros(int(nintervals))

    for ii in range(0, int(nintervals)):

        # define indices of interval limits
        mnind = np.argmin(np.abs(t - (t[0] + ii*dt)))
        mxind = np.argmin(np.abs(t - (t[0] + (ii+1)*dt)))

        # new truncated vectors
        tt0 = t[mnind:mxind]
        hh0 = d[mnind:mxind]
        hmean = np.nanmean(hh0)

        hh = hh0[~np.isnan(hh0)]
        tt = tt0[~np.isnan(hh0)]

        F, Pxx = signal.welch(signal.detrend(hh), fs=fs, window=winhann, nperseg=nfft, \
                       noverlap=int(np.floor(nfft*overlap)), nfft=nfft)

        krad = implicit_wavenumber(F, hmean)

        Phh = atten_correct(Pxx, F, krad, hmean)

#        Hs[ii], Hs_infr[ii], Hs_swell[ii], Hs_wind[ii], Tmean[ii], \
#                    Tmean_infr[ii], Tmean_swell[ii], Tmean_wind[ii], \
#                    Tp[ii], L[ii] = compute_wavestats(Phh, F, hmean)

        waves = compute_wavestats(Phh, F, hmean)
        Hs[ii] = waves['Hs']
        Hs_swell[ii] = waves['Hs_swell']
        Hs_wind[ii] = waves['Hs_wind']
        Tmean[ii] = waves['Tmean']
        Tmean_swell[ii] = waves['Tmean_swell']
        Tmean_wind[ii] = waves['Tmean_wind']
        Tp[ii] = waves['Tp']
        L[ii] = waves['wavelength']
        steepness[ii] = waves['steepness']
        M[ii] = waves['Miche']
        xi_b[ii] = waves['Iribarren']
        eps_sc[ii] = waves['surf_scaling']


        # new corresponding time and depth vectors
        timevec[ii] = np.mean(tt)
        depth[ii] = np.mean(hh)

    plt.figure(1)
    plt.plot(timevec, Hs)

    plt.figure(2)
    plt.plot(timevec, Tp)

    rf = 0.1*xi_b**2
    rf[rf > 1] = 1
    M[M > 1] = 1

    waves = {"yearday": timevec,
             "depth": depth,
             "Hs": Hs,
             "Hs_swell": Hs_swell,
             "Hs_wind": Hs_wind,
             "Tmean": Tmean,
             "Tmean_swell": Tmean_swell,
             "Tmean_wind": Tmean_wind,
             "Tp": Tp,
             "wavelength": L,
             "steepness": steepness,
             "Miche": M,
             "Iribarren": xi_b,
             "surf_scaling": eps_sc,}

    # save new variables
    fout = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", \
                     "processed","pressure", "wavestats", fn)
    # np.save(fout, waves)
