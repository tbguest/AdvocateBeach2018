


import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal

drivechar = '/media/tristan2/Advocate2018_backup2'


%matplotlib qt5

targfile = os.path.join(drivechar, 'data','raw','images','OutdoorValidation','IMG_0206.JPG')
# targfile = os.path.join(drivechar, 'data','raw','images','BeachSurveys','18_10_2018','AM','dense_array1','IMG_1467.JPG')

im = plt.imread(targfile)
plt.imshow(im)
print("Click on targets, top to bottom.")
x = plt.ginput(2)
print("clicked", x)
plt.show()
dpix = np.sqrt(np.abs(x[0][0] - x[1][0])**2 + np.abs(x[0][1] - x[1][1])**2)

# pixels/m
d1 = 163 # mm
# d1 = 155 # mm
# d1=600

scaling1 = d1/dpix
