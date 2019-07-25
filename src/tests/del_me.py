import numpy as np
import matplotlib.pyplot as plt

%matplotlib qt5

vvv = np.load('/home/tristan2/Projects/AdvocateBeach2018/data/interim/range_data/bed_level/tide19/sonar74.npy', allow_pickle=True).item()

smth = vvv['smoothed chunks']
rtime = smth[0]
sonar_range = smth[1]

raw = vvv['raw range']
t = raw[0]
r = raw[1]

plt.plot(t,r)
plt.plot(rtime, sonar_range,'.')
