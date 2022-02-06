import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys

sys.path.append(".")
from src.visualization.plot_beach_profile_data import save_figures

# %matplotlib qt5

# change default font size
plt.rcParams.update({"font.size": 12})

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~")  # linux

drivechar = "/media/tristan/Advocate2018_backup2"

# figsdn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "reports", "figures")
figsdn = os.path.join(
    homechar,
    "Documents",
    "manuscripts",
    "guest-hay-jmse-2022",
    "src",
    "figures",
    "revised",
)

start_tide = 1
tide_range = range(start_tide, 28)

beta_slope = 0.12
iribarren_crit = np.abs((np.pi) ** 3 / (2 * beta_slope)) ** (1 / 4)

Hs = []
Tp = []
steepness = []
iribarren = []
wave_energy = []
wave_energy_wind = []
wave_energy_swell = []
depth = []
yearday = []
tide_count = []
miche = []

yearday0 = []
Hs0 = []
Tp0 = []
steepness0 = []
iribarren0 = []
miche0 = []
depth0 = []

# # breaker height estimation params
# kap = 0.8
# g = 9.81
#
# Hb = kap


counter = 0

for tide in tide_range:
    # tide = 15

    counter += 1

    # wavefn = os.path.join(
    #     homechar,
    #     "Projects",
    #     "AdvocateBeach2018",
    #     "data",
    #     "processed",
    #     "pressure",
    #     "wavestats",
    #     "tide" + str(tide) + ".json",
    # )
    wavefn = os.path.join(
        drivechar,
        "data",
        "processed",
        "pressure",
        "wavestats",
        "tide" + str(tide) + ".json",
    )

    # wavefn = os.path.join(
    #     drivechar,
    #     "data",
    #     "processed",
    #     "pressure",
    #     "wavestats",
    #     "tide" + str(tide) + ".npy",
    # )

    # wave data
    if not os.path.exists(wavefn):
        continue

    # bar = np.load(wavefn, allow_pickle=True).item()
    with open(wavefn, "r") as fpp:
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
    miche.extend(np.array(bar["Miche"]))
    depth.extend(np.array(bar["depth"]))

    tide_count.append(counter)

    yearday0.append(np.array(bar["yearday"])[0])
    Hs0.append(np.array(bar["Hs"])[0])
    Tp0.append(np.array(bar["Tp"])[0])
    steepness0.append(np.array(bar["steepness"])[0])
    iribarren0.append(np.array(bar["Iribarren"])[0])
    miche0.append(np.array(bar["Miche"])[0])
    depth0.append(np.array(bar["depth"])[0])

fig1, ax1 = plt.subplots(nrows=4, ncols=1, figsize=(8, 6), num="wavedata")
ax1[0].plot(yearday, depth, ".")
# ax1[0].plot(yearday0, depth0, "r.")
for n in range(len(tide_count)):
    ax1[0].text(yearday0[n], 7.5, str(tide_count[n]))
ax1[0].text(295, 9.5, "tide")
ax1[0].text(291, 5.96, "a")
ax1[0].set_ylabel("$h$ [m]")
ax1[0].set_xticklabels([])
ax1[0].set_xlim(np.min(yearday), np.max(yearday))
ax1[0].tick_params(direction="in", top=1, right=1)
ax1[1].plot(yearday, Hs, ".")
# ax1[1].plot(yearday0, Hs0, "r.")
ax1[1].set_ylabel("$H_s$ [m]")
ax1[1].set_xticklabels([])
ax1[1].set_xlim(np.min(yearday), np.max(yearday))
ax1[1].tick_params(direction="in", top=1, right=1)
ax1[1].text(291, 1.5, "b")
ax1[2].plot(yearday, Tp, ".")
# ax1[2].plot(yearday0, Tp0, "r.")
ax1[2].set_ylabel("$T_p$ [s]")
ax1[2].set_xticklabels([])
ax1[2].set_xlim(np.min(yearday), np.max(yearday))
ax1[2].tick_params(direction="in", top=1, right=1)
ax1[2].text(291, 10.5, "c")
ax1[3].plot(yearday, steepness, ".")
# ax1[3].plot(yearday0, steepness0, "r.")
ax1[3].set_ylabel("$H_0/L_0$")
ax1[3].plot(yearday, 0.01 * np.ones(len(yearday)), "k--")
# ax1[3].set_xticklabels([])
ax1[3].set_xlim(np.min(yearday), np.max(yearday))
ax1[3].tick_params(direction="in", top=1, right=1)
ax1[3].set_xlabel("yearday 2018")
ax1[3].text(291, 0.055, "d")

# ax1[4].plot(yearday, iribarren, '.')
# ax1[4].plot(yearday0, iribarren0, 'r.')
# ax1[4].set_ylabel(r'$\xi_0$')
# ax1[4].set_xlabel('yearday 2018')
# ax1[4].set_xlim(np.min(yearday), np.max(yearday))
# ax1[4].tick_params(direction='in',top=1,right=1)
fig1.tight_layout()

fig2, ax2 = plt.subplots(nrows=5, ncols=1, figsize=(8, 6), num="wavedata 2")
ax2[0].plot(yearday, Hs, ".")
ax2[0].plot(yearday0, Hs0, "r.")
for n in range(len(tide_count)):
    ax2[0].text(yearday0[n], 2, str(tide_count[n]))
ax2[0].text(295, 2.45, "tide")
ax2[0].set_ylabel("$H_s$ [m]")
ax2[0].set_xticklabels([])
ax2[0].set_xlim(np.min(yearday), np.max(yearday))
ax2[0].tick_params(direction="in", top=1, right=1)
ax2[1].plot(yearday, Tp, ".")
ax2[1].plot(yearday0, Tp0, "r.")
ax2[1].set_ylabel("$T_p$ [s]")
ax2[1].set_xticklabels([])
ax2[1].set_xlim(np.min(yearday), np.max(yearday))
ax2[1].tick_params(direction="in", top=1, right=1)
ax2[2].plot(yearday, steepness, ".")
ax2[2].plot(yearday0, steepness0, "r.")
ax2[2].set_ylabel("$H_0/L_0$")
ax2[2].plot(yearday, 0.01 * np.ones(len(yearday)), "k--")
ax2[2].set_xticklabels([])
ax2[2].set_xlim(np.min(yearday), np.max(yearday))
ax2[2].tick_params(direction="in", top=1, right=1)
ax2[3].plot(yearday, miche, ".")
ax2[3].plot(yearday0, miche0, "r.")
ax2[3].set_ylabel("$R^{2}$")
ax2[3].set_xticklabels([])
ax2[3].set_xlim(np.min(yearday), np.max(yearday))
ax2[3].set_yscale("log")
ax2[3].set_ylim(1 * 10 ** -3, 10 ** 0)
ax2[3].tick_params(direction="in", top=1, right=1)
ax2[4].plot(yearday, iribarren, ".")
ax2[4].plot(yearday0, iribarren0, "r.")
ax2[4].set_ylabel(r"$\xi_0$")
ax2[4].set_xlabel("yearday 2018")
ax2[4].set_xlim(np.min(yearday), np.max(yearday))
ax2[4].tick_params(direction="in", top=1, right=1)
fig2.tight_layout()


fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(8, 4.5), num="wavedata3")
ax3[0].plot(yearday, np.array(depth) - 1.77, ".")
# ax3[0].plot(yearday0, depth0, "r.")
for n in range(len(tide_count)):
    ax3[0].text(yearday0[n], 7.5 - 1.77, str(tide_count[n]))
ax3[0].text(295, 9.5 - 1.77, "tide")
ax3[0].text(291, 5.96 - 1.77, "a")
ax3[0].set_ylabel("$h$ [m]")
ax3[0].set_xticklabels([])
ax3[0].set_xlim(np.min(yearday), np.max(yearday))
ax3[0].tick_params(direction="in", top=1, right=1)
ax3[1].plot(yearday, Hs, ".")
# ax3[1].plot(yearday0, Hs0, "r.")
ax3[1].set_ylabel("$H_s$ [m]")
ax3[1].set_xticklabels([])
ax3[1].set_xlim(np.min(yearday), np.max(yearday))
ax3[1].tick_params(direction="in", top=1, right=1)
ax3[1].text(291, 1.5, "b")
ax3[2].plot(yearday, Tp, ".")
# ax3[2].plot(yearday0, Tp0, "r.")
ax3[2].set_ylabel("$T_p$ [s]")
# ax3[2].set_xticklabels([])
ax3[2].set_xlim(np.min(yearday), np.max(yearday))
ax3[2].tick_params(direction="in", top=1, right=1)
ax3[2].text(291, 10.5, "c")
ax3[2].set_xlabel("yearday 2018")
fig3.tight_layout()


R_mean = np.nanmean(miche)

# plt.show()

saveFlag = 1

# EXPORT PLOTS
if saveFlag == 1:

    # savedn = os.path.join(figsdn, "wave_stats")
    savedn = os.path.join(figsdn, "wave_stats_basic")

    save_figures(savedn, "wave_stats", fig1)

