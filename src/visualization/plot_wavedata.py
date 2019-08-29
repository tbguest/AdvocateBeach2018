import numpy as np
import matplotlib.pyplot as plt
import os
import json

from src.visualization.plot_beach_profile_data import save_figures

# change default font size
plt.rcParams.update({'font.size': 12})

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~") # linux

figsdn = os.path.join(homechar,'Projects','AdvocateBeach2018',\
'reports','figures')

start_tide = 1
tide_range = range(start_tide, 28)

beta_slope = 0.12
iribarren_crit = np.abs((np.pi)**3/(2*beta_slope))**(1/4)

Hs = []
Tp = []
steepness = []
iribarren = []
wave_energy = []
wave_energy_wind = []
wave_energy_swell = []
maxdepth = []
yearday = []
tide_count = []

yearday0 = []
Hs0 = []
Tp0 = []
steepness0 = []
iribarren0 = []

counter = 0

for tide in tide_range:
# tide = 15

    counter += 1

    wavefn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "processed", \
                  "pressure", "wavestats", "tide" + str(tide) + ".json")

    # wave data
    if not os.path.exists(wavefn):
        continue


    # bar = np.load(wavefn, allow_pickle=True).item()
    with open(wavefn, 'r') as fpp:
        bar = json.load(fpp)
    # yearday.append(np.mean(np.array(bar["yearday"])))
    # Hs.append(np.mean(np.array(bar["Hs"])))
    # Tp.append(np.mean(np.array(bar["Tp"])))
    # steepness.append(np.mean(np.array(bar["steepness"])))
    # iribarren.append(np.mean(np.array(bar["Iribarren"])))
    # wave_energy.append(np.mean(np.array(bar["wave_energy"])))
    # wave_energy_wind.append(np.mean(np.array(bar["wave_energy_wind"])))
    # wave_energy_swell.append(np.mean(np.array(bar["wave_energy_swell"])))
    # maxdepth.append(np.max(bar["depth"]))

    yearday.extend(np.array(bar["yearday"]))
    Hs.extend(np.array(bar["Hs"]))
    Tp.extend(np.array(bar["Tp"]))
    steepness.extend(np.array(bar["steepness"]))
    iribarren.extend(np.array(bar["Iribarren"]))
    wave_energy.extend(np.array(bar["wave_energy"]))
    wave_energy_wind.extend(np.array(bar["wave_energy_wind"]))
    wave_energy_swell.extend(np.array(bar["wave_energy_swell"]))
    # maxdepth.extend(np.max(bar["depth"]))

    tide_count.append(counter)

    yearday0.append(np.array(bar["yearday"])[0])
    Hs0.append(np.array(bar["Hs"])[0])
    Tp0.append(np.array(bar["Tp"])[0])
    steepness0.append(np.array(bar["steepness"])[0])
    iribarren0.append(np.array(bar["Iribarren"])[0])

fig1, ax1 = plt.subplots(nrows=4, ncols=1, figsize=(8,6), num='wavedata')
ax1[0].plot(yearday, Hs, '.')
ax1[0].plot(yearday0, Hs0, 'r.')
for n in range(len(tide_count)):
    ax1[0].text(yearday0[n], 2, str(tide_count[n]))
ax1[0].text(295,2.45,'tide')
ax1[0].set_ylabel('$H_s$ [m]')
ax1[0].set_xticklabels([])
ax1[0].set_xlim(np.min(yearday), np.max(yearday))
ax1[0].tick_params(direction='in',top=1,right=1)
ax1[1].plot(yearday, Tp, '.')
ax1[1].plot(yearday0, Tp0, 'r.')
ax1[1].set_ylabel('$T_p$ [s]')
ax1[1].set_xticklabels([])
ax1[1].set_xlim(np.min(yearday), np.max(yearday))
ax1[1].tick_params(direction='in',top=1,right=1)
ax1[2].plot(yearday, steepness, '.')
ax1[2].plot(yearday0, steepness0, 'r.')
ax1[2].set_ylabel('$H_0/L_0$')
ax1[2].plot(yearday, 0.01*np.ones(len(yearday)), 'k--')
ax1[2].set_xticklabels([])
ax1[2].set_xlim(np.min(yearday), np.max(yearday))
ax1[2].tick_params(direction='in',top=1,right=1)
ax1[3].plot(yearday, iribarren, '.')
ax1[3].plot(yearday0, iribarren0, 'r.')
ax1[3].set_ylabel(r'$\xi_0$')
ax1[3].set_xlabel('yearday 2018')
ax1[3].set_xlim(np.min(yearday), np.max(yearday))
ax1[3].tick_params(direction='in',top=1,right=1)
fig1.tight_layout()


saveFlag = 1

# EXPORT PLOTS
if saveFlag == 1:

    savedn = os.path.join(figsdn,'wave_stats')

    save_figures(savedn, 'wave_stats', fig1)
