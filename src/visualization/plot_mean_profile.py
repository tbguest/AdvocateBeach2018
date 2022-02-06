import numpy as np
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import math
import numpy.matlib
import sys
import matplotlib.lines as mlines
import matplotlib

sys.path.append(".")
# from .data.regresstools import lowess
from src.visualization.plot_beach_profile_data import save_figures

plt.rcParams.update({"font.size": 14})
# font = {"family": "normal", "size": 26}

# matplotlib.rc("font", **font)

force_y_offset = 0


saveFlag = True

# 2018/10/14;07:09:00;9.9(m);32.5(ft);1
# 2018/10/14;19:26:00;10(m);32.8(ft);2
# 2018/10/15;07:58:00;9.4(m);30.8(ft);3
# 2018/10/15;20:15:00;9.5(m);31.2(ft);4
# 2018/10/16;08:50:00;9(m);29.5(ft);5
# 2018/10/16;21:09:00;9(m);29.5(ft);6
# 2018/10/17;09:48:00;8.7(m);28.5(ft);7
# 2018/10/17;22:07:00;8.7(m);28.5(ft);8
# 2018/10/18;10:48:00;8.5(m);27.9(ft);9
# 2018/10/18;23:08:00;8.6(m);28.2(ft);10
# 2018/10/19;11:47:00;8.6(m);28.2(ft);11
# 2018/10/20;00:06:00;8.7(m);28.5(ft);12
# 2018/10/20;12:41:00;8.8(m);28.9(ft);13
# 2018/10/21;00:58:00;8.9(m);29.2(ft);14
# 2018/10/21;13:28:00;9.1(m);29.9(ft);15
# 2018/10/22;01:44:00;9.2(m);30.2(ft);16
# 2018/10/22;14:09:00;9.5(m);31.2(ft);17
# 2018/10/23;02:26:00;9.5(m);31.2(ft);18
# 2018/10/23;14:47:00;9.9(m);32.5(ft);19
# 2018/10/24;03:05:00;9.9(m);32.5(ft);20
# 2018/10/24;15:24:00;10.3(m);33.8(ft);21
# 2018/10/25;03:44:00;10.2(m);33.5(ft);22
# 2018/10/25;16:02:00;10.6(m);34.8(ft);23
# 2018/10/26;04:23:00;10.4(m);34.1(ft);24
# 2018/10/26;16:41:00;10.9(m);35.8(ft);25
# 2018/10/27;05:06:00;10.6(m);34.8(ft);26
# 2018/10/27;17:24:00;11.1(m);36.4(ft);27

tidehis = [
    9.9,
    10,
    9.4,
    9.5,
    9,
    9,
    8.7,
    8.7,
    8.5,
    8.6,
    8.6,
    8.7,
    8.8,
    8.9,
    9.1,
    9.2,
    9.5,
    9.5,
    9.9,
    9.9,
    10.3,
    10.2,
    10.6,
    10.4,
    10.9,
    10.6,
    11.1,
]

meanTideRange = np.mean(tidehis)

# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide19/array_position1.npy"
arrayStations19 = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide19/array_position2.npy"
# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide19/array_position3.npy"
# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide19/array_position4.npy"
# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide19/array_position5.npy"

# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide27/array_position1.npy"
arrayStations27 = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide27/array_position2.npy"
# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide27/array_position3.npy"
# arrayStations = "/media/tristan/Advocate2018_backup2/data/processed/GPS/array/tide27/array_position4.npy"

loc19 = np.load(arrayStations19, allow_pickle=True)
loc27 = np.load(arrayStations27, allow_pickle=True)


# change default font size
plt.rcParams.update({"font.size": 12})

grid_spec1 = "zline"
# grid_spec = "longshore2"
# grid_spec = "longshore1"
grid_spec2 = "dense_array2"

# for portability
# homechar = "C:\\"
homechar = os.path.expanduser("~")  # linux
drivechar = "/media/tristan/Advocate2018_backup2"

figsdn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "reports", "figures")

hwl = [-21, -9, -15, -15, -18, -15, -15, -15, -18, -21, -18, -18, -18, -18]
# hwl = [-9,-15,-15,-18,-15,-15,-15,-18,-21,-18,-18,-18,-18]

# if grid_spec == "cross_shore":
#     start_tide = 13
#     # start_tide = 14
#     # hwl = hwl[1:]
# elif grid_spec == "longshore1":
#     start_tide = 15
#     hwl = hwl[2:]
# else:
#     start_tide = 14
#     hwl = hwl[1:]

tides = np.arange(14, 28)

fig, ax = plt.subplots(1, 1, figsize=(7, 3))

# for inset
# fig, ax = plt.subplots(1, 1, figsize=(7 / 1.5, 3 / 1.5))


yall = []
zall = []

for t in tides:
    tide = "tide" + str(t)

    gpsfn1 = os.path.join(
        drivechar, "data", "interim", "GPS", "by_tide", tide, grid_spec1 + ".npy"
    )
    gpsfn2 = os.path.join(
        drivechar, "data", "interim", "GPS", "by_tide", tide, grid_spec2 + ".npy"
    )

    foo1 = np.load(gpsfn1, allow_pickle=True).item()
    foo2 = np.load(gpsfn2, allow_pickle=True).item()

    yround = np.round(foo1["y"]).astype(int)
    yall.extend(yround)
    zall.extend(foo1["z"])

    # fig, ax = plt.subplots(1, 1)
    ax.plot(yround + force_y_offset, foo1["z"], "C0-", linewidth=0.75)


yvec0 = np.arange(np.min(yall), np.max(yall), 3)
zmean = np.zeros((1, len(yvec0)))[0]
yall = np.array(yall)
zall = np.array(zall)

yvec = yvec0 + force_y_offset
yall = yall + force_y_offset
yround = yround + force_y_offset

for i in range(len(yvec)):
    zmean[i] = np.mean(zall[yall == int(yvec[i])])

i_zhwl = np.argmin(np.abs(yvec0 - np.mean(hwl)))
zhwl = zmean[i_zhwl]

# force profile artificially further seaward
# yvec = np.append(yvec, np.array([75, 78, 81, 84]))
zmean = np.append(zmean, np.array([-4.42, -4.75, -5.08, -5.41]) - 0.12 * 18)
yvec = np.append(yvec, np.array([75, 78, 81, 84])) - force_y_offset


yHW = np.arange(0, 100, 1)
yLW = np.arange(78.9, 100, 1)
ax.plot(yvec, zmean, "k-", linewidth=2)
# ax.plot(yvec, zhwl * np.ones(len(yvec)), "k--")
# ax.plot(yvec, zhwl * np.ones(len(yvec)) - meanTideRange, "k--")  # 78.9
ax.plot(yHW - 18, zhwl * np.ones(len(yHW)), "k--")
ax.plot(yLW - 18, zhwl * np.ones(len(yLW)) - meanTideRange, "k--")  # 78.9
ax.text(87 - 18, zhwl + 0.2, "MHW")
ax.text(87 - 18, zhwl - meanTideRange + 0.2, "MLW")
ax.plot(58 - 18, -2, "C2.", markersize=12)
ax.text(59 - 18, -1.75, "PT")

ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
# ax.set_xlim([np.min(yvec), np.max(yvec)])
ax.set_xlim([-12 - 18, 96 - 18])
ax.set_ylim([-5.4, 7])
ax.tick_params(axis="both", direction="in", top=True, right=True)

# for inset
# ax.set_xlim([-27.5, -5])
# ax.set_ylim([2.8, 6.3])

# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

ax.set_ylabel("elevation [m]")
ax.set_xlabel("cross-shore coordinate [m]")
fig.tight_layout()
plt.show()


fig2, ax2 = plt.subplots(1, 1, figsize=(7 / 1.5, 3 / 1.5))

for t in tides:
    tide = "tide" + str(t)

    gpsfn1 = os.path.join(
        drivechar, "data", "interim", "GPS", "by_tide", tide, grid_spec1 + ".npy"
    )
    gpsfn2 = os.path.join(
        drivechar, "data", "interim", "GPS", "by_tide", tide, grid_spec2 + ".npy"
    )

    foo1 = np.load(gpsfn1, allow_pickle=True).item()
    foo2 = np.load(gpsfn2, allow_pickle=True).item()

    # fig, ax = plt.subplots(1, 1)
    ax2.plot(yround[0:13], foo1["z"][0:13] - zmean[0:13], "C0-", linewidth=0.75)

# ax.plot(yvec, zmean, "k-", linewidth=2)
ax2.plot(yvec, zhwl * np.ones(len(yvec)), "k--")
ax2.plot(yvec, zhwl * np.ones(len(yvec)) - meanTideRange, "k--")
# ax2.text(30, zhwl + 0.2, "MHW")
# ax2.text(30, zhwl - meanTideRange + 0.2, "MLW")

ax2.tick_params(axis="y", direction="in")
ax2.tick_params(axis="x", direction="in")
ax2.set_xlim([np.min(yvec), np.max(yvec)])
ax2.tick_params(axis="both", direction="in", top=True, right=True)

# for inset
# ax2.set_xlim([-27.5 + force_y_offset, -5 + force_y_offset])
ax2.set_xlim([-27.5 + force_y_offset, -5])
ax2.set_ylim([-0.25, 0.25])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax2.set_ylabel("$z - \overline{z}$ [m]", fontsize=14)
ax2.set_xlabel("cross-shore coordinate [m]", fontsize=14)
fig2.tight_layout()

plt.show()

savedn = "/home/tristan/Documents/manuscripts/guest-hay-jmse-2022/src/figures/revised"
if saveFlag:
    save_figures(savedn, "mean_profile", fig)
    save_figures(savedn, "mean_profile_inset", fig2)


"""

Section 2

"""

## testing

"""
1252,357933.0300,5022715.2403,4.2620,NTARG1,HRMS:0.005,VRMS:0.005,STATUS:FIXED,SATS:11,PDOP:1.460,HDOP:0.842,VDOP:1.192,TDOP:0.839,GDOP:1.683,DATE:01-01-2010,TIME:05:01:32,RODHGT2:1.471
1253,357932.8343,5022713.9053,4.1869,CTARG1,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:11,PDOP:1.462,HDOP:0.843,VDOP:1.195,TDOP:0.841,GDOP:1.687,DATE:01-01-2010,TIME:05:01:50,RODHGT2:1.471
1254,357934.2225,5022714.1795,4.2730,STARG1,HRMS:0.006,VRMS:0.006,STATUS:FIXED,SATS:11,PDOP:1.465,HDOP:0.844,VDOP:1.197,TDOP:0.843,GDOP:1.690,DATE:01-01-2010,TIME:05:02:11,RODHGT2:1.471

1263,357930.7046,5022711.6730,3.7394,NTARG2,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:11,PDOP:1.617,HDOP:0.800,VDOP:1.405,TDOP:1.261,GDOP:2.051,DATE:01-01-2010,TIME:05:28:52,RODHGT2:1.471
1264,357930.5826,5022710.5904,3.5734,CTARG2,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:13,PDOP:1.617,HDOP:0.800,VDOP:1.405,TDOP:1.261,GDOP:2.051,DATE:01-01-2010,TIME:05:29:13,RODHGT2:1.471
1265,357931.7242,5022710.8105,3.7184,STARG2,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:13,PDOP:1.617,HDOP:0.800,VDOP:1.405,TDOP:1.261,GDOP:2.051,DATE:01-01-2010,TIME:05:29:30,RODHGT2:1.471

1266,357927.9902,5022708.4104,3.1774,NTARG3,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:11,PDOP:1.669,HDOP:0.900,VDOP:1.405,TDOP:1.192,GDOP:2.051,DATE:01-01-2010,TIME:05:40:20,RODHGT2:1.471
1267,357928.2278,5022707.4896,3.1164,CTARG3,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:11,PDOP:1.669,HDOP:0.900,VDOP:1.405,TDOP:1.192,GDOP:2.051,DATE:01-01-2010,TIME:05:40:36,RODHGT2:1.471
1268,357929.3607,5022707.4440,3.2034,STARG3,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:11,PDOP:1.669,HDOP:0.900,VDOP:1.405,TDOP:1.192,GDOP:2.051,DATE:01-01-2010,TIME:05:40:53,RODHGT2:1.471

1273,357924.5323,5022707.1940,2.7924,NTARG4,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:10,PDOP:1.617,HDOP:0.800,VDOP:1.405,TDOP:1.261,GDOP:2.051,DATE:01-01-2010,TIME:06:03:29,RODHGT2:1.471
1274,357924.0909,5022705.8421,2.6544,CTARG4,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:10,PDOP:1.725,HDOP:1.000,VDOP:1.405,TDOP:1.109,GDOP:2.051,DATE:01-01-2010,TIME:06:03:49,RODHGT2:1.471
1275,357925.6879,5022706.5329,2.8244,STARG4,HRMS:0.005,VRMS:0.006,STATUS:FIXED,SATS:10,PDOP:1.617,HDOP:0.800,VDOP:1.405,TDOP:1.261,GDOP:2.051,DATE:01-01-2010,TIME:06:04:10,RODHGT2:1.471
"""

# PT frame
# 357897.9616,5022673.2420,-2.0021

ptcoords = np.array([357897.9616, 5022673.2420])
# rotated: y, x = (39.75223036547932, -26.083099776906725)

# UTM coords of origin (Adv2015)
originx = 3.579093296000000e05
originy = 5.022719408400000e06

# plt.figure()
# plt.plot(357899.3979000000 - originx,5022717.801400000 - originy, 'ro')
# plt.plot(357908.4006000000 - originx, 5022709.436700001 - originy, 'rx')
# plt.plot(originx - originx, originy - originy, 'ko')
# plt.plot(357933.0300 - originx,5022715.2403 - originy, 'b.')
# plt.plot(357932.8343 - originx,5022713.9053 - originy, 'b.')
# plt.plot(357934.2225 - originx,5022714.1795 - originy, 'b.')

# plt.plot(357924.5323 - originx,5022707.1940 - originy, 'bx')
# plt.plot(357924.0909 - originx,5022705.8421 - originy, 'bx')
# plt.plot(357925.6879 - originx,5022706.5329 - originy, 'bx')

# plt.show()


ptcoordsp = [ptcoords[1] - originy, ptcoords[0] - originx]

s1 = np.array(
    [
        [357933.0300, 5022715.2403],
        [357932.8343, 5022713.9053],
        [357934.2225, 5022714.1795],
    ]
)
s1p = [np.mean(s1[:, 1]) - originy, np.mean(s1[:, 0]) - originx]

s2 = np.array(
    [
        [357930.7046, 5022711.6730],
        [357930.5826, 5022710.5904],
        [357931.7242, 5022710.8105],
    ]
)
s2p = [np.mean(s2[:, 1]) - originy, np.mean(s2[:, 0]) - originx]

s3 = np.array(
    [
        [357927.9902, 5022708.4104],
        [357928.2278, 5022707.4896],
        [357929.3607, 5022707.4440],
    ]
)
s3p = [np.mean(s3[:, 1]) - originy, np.mean(s3[:, 0]) - originx]

s4 = np.array(
    [
        [357924.5323, 5022707.1940],
        [357924.0909, 5022705.8421],
        [357925.6879, 5022706.5329],
    ]
)
s4p = [np.mean(s4[:, 1]) - originy, np.mean(s4[:, 0]) - originx]

# rotation angle from 2015 stake coordinates
theta_rot = math.pi + math.atan(
    (357899.3979000000 - 357908.4006000000) / (5022717.801400000 - 5022709.436700001)
)

# rotation matrix
R = np.array(
    [
        math.cos(theta_rot),
        -math.sin(theta_rot),
        math.sin(theta_rot),
        math.cos(theta_rot),
    ]
).reshape((2, 2))

# apply to coordinates
# Ymat1 = np.array([s1North, s1East])
y1, x1 = np.matmul(R, s1p)
y2, x2 = np.matmul(R, s2p)
y3, x3 = np.matmul(R, s3p)
y4, x4 = np.matmul(R, s4p)
z1 = 4.2620
z2 = 3.7184
z3 = 3.1164
z4 = 2.7924

pty, ptx = np.matmul(R, ptcoordsp)

######

tides = ["18", "19"]
# tides = ["26", "27"]

fig, ax = plt.subplots(1, 1, figsize=(7, 3))

counter = 0

for t in tides:
    tide = "tide" + str(t)

    # tide_range = range(start_tide, 28)
    # tide_axis = np.arange(start_tide + 1, 28)  # for plotting later

    gpsfn1 = os.path.join(
        drivechar, "data", "interim", "GPS", "by_tide", tide, grid_spec1 + ".npy"
    )
    gpsfn2 = os.path.join(
        drivechar, "data", "interim", "GPS", "by_tide", tide, grid_spec2 + ".npy"
    )

    foo1 = np.load(gpsfn1, allow_pickle=True).item()
    foo2 = np.load(gpsfn2, allow_pickle=True).item()

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(foo1["y"], foo1["z"])

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(foo2["y"], foo2["z"])
    # plt.show()

    xz = foo1["x"]
    yz = foo1["y"]
    zz = foo1["z"]
    x = foo2["x"].reshape(6, 24)
    y = foo2["y"].reshape(6, 24)
    z = foo2["z"].reshape(6, 24)

    # plt.contourf(x, y, z)
    # plt.show()

    tmpx = np.arange(-30, -10)
    yzz, xzz = np.meshgrid(yz, tmpx)

    # plt.figure()
    # plt.plot(xzz, yzz, ".")
    # plt.show()

    zzz = numpy.matlib.repmat(zz, len(tmpx), 1)

    # plt.contourf(xzz, yzz, zzz, levels=len(yz))
    # plt.plot(loc19.item()["x"], loc19.item()["y"], "b.")
    # plt.plot(loc27.item()["x"], loc27.item()["y"], "b.")
    # # plt.show()

    # plt.contourf(x, y, z)
    # plt.plot(loc19.item()["x"], loc19.item()["y"], "b.")
    # plt.plot(loc27.item()["x"], loc27.item()["y"], "b.")
    # plt.plot(-x1, y1, "r.")
    # plt.plot(-x2, y2, "r.")
    # plt.plot(-x3, y3, "r.")
    # plt.plot(-x4, y4, "r.")
    # # plt.show()

    yshort = np.round(y[:, -1]).astype(int)
    zshort = z[:, -1]
    ylong = np.round(yz).astype(int)
    zlong = zz

    iylong_cut = np.hstack(
        (np.argwhere(ylong < -17).flatten(), np.argwhere(ylong > -7).flatten())
    )
    ylong_cut = list(ylong[iylong_cut])
    zlong_cut = list(zlong[iylong_cut])
    for yi in yshort[::-1]:
        ylong_cut.insert(5, yi)
    for zi in zshort[::-1]:
        zlong_cut.insert(5, zi)

    composite_y = np.array(ylong_cut)
    composite_z = np.array(zlong_cut)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(y[:, -1], z[:, -1])
    # ax.plot(yz, zz)
    if counter == 0:
        ax.plot(composite_y + force_y_offset, composite_z, "C0", linewidth=1.5)
    else:
        ax.plot(composite_y + force_y_offset, composite_z, "k", linewidth=2)
    if tide == "tide18" or tide == "tide19":
        (h_array,) = ax.plot(
            loc19.item()["y"] + force_y_offset,
            loc19.item()["z"] + 1.2,
            "b.",
            markersize=12,
        )
        (h_station,) = ax.plot(y1 + force_y_offset, z1, "r.", markersize=12)
        ax.plot(y2 + force_y_offset, z2, "r.", markersize=12)
        ax.plot(y3 + force_y_offset, z3, "r.", markersize=12)
        ax.plot(y4 + force_y_offset, z4, "r.", markersize=12)
    elif tide == "tide26" or tide == "tide27":
        h_array = ax.plot(
            loc27.item()["y"] + force_y_offset,
            loc27.item()["z"] + [0.4, 0.4, 0, 0] + 1.2,
            "b.",
            markersize=12,
        )
    ax.tick_params(axis="both", direction="in", top=True, right=True)

    counter += 1


if tide == "tide18" or tide == "tide19":
    i_zhwl = np.argmin(np.abs(composite_y - -15))
elif tide == "tide26" or tide == "tide27":
    i_zhwl = np.argmin(np.abs(composite_y - -force_y_offset))
zhwl = zmean[i_zhwl]

# ax.grid()
if tide == "tide18" or tide == "tide19":
    jnky = np.arange(6.8, composite_y[-1] + force_y_offset, 0.1)
    # ax.plot(composite_y + force_y_offset, zhwl * np.ones(len(composite_y)), "k--")
    ax.plot(jnky, zhwl * np.ones(len(jnky)), "k--")
    ax.text(16, zhwl + 0.2, "HW")
elif tide == "tide26" or tide == "tide27":
    jnky = np.arange(3.6, composite_y[-1] + force_y_offset, 0.1)
    # ax.plot(composite_y + force_y_offset, zhwl * np.ones(len(composite_y)) - 0.3, "k--")
    ax.plot(jnky, zhwl * np.ones(len(jnky)) - 0.3, "k--")
    ax.text(16, zhwl + 0.2 - 0.3, "HW")
# ax.text(0 + force_y_offset, zhwl + 0.05, "HW")
ax.set_ylabel("elevation [m]")
# ax.set_xlabel("cross-shore coordinate [m]")
# ax.set_xlim([np.min(composite_y) + 5 + force_y_offset, 3 + force_y_offset])
# ax.set_xlim([-6, 16])
ax.set_xlim([-1.5, 17.5])
ax.set_ylim([1.95, 6.75])
if tide == "tide18" or tide == "tide19":
    plt.legend((h_array, h_station), ("array cameras", "overhead frame stations"))
elif tide == "tide26" or tide == "tide27":
    # plt.legend(h_array, ["array cameras"])
    pass
# blue_dot = mlines.Line2D(
#     [], [], color="b", marker=".", markersize=12, label="array cameras"
# )
# red_dot = mlines.Line2D(
#     [], [], color="r", marker=".", markersize=12, label="overhead frame stations"
# )
# plt.legend(handles=[blue_dot, red_dot])

fig.tight_layout()
plt.show()

savedn = "/home/tristan/Documents/manuscripts/guest-hay-jmse-2022/src/figures/revised"
if saveFlag:
    save_figures(savedn, f"{tide}_profile_nolegend", fig)
