
import numpy as np
from random import gauss
from random import random
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




# series1 = [gauss(0.0, 1.0) for i in range(10000)]
series1 = [random() for i in range(10000)]

plt.figure(1)
plt.hist(series1)

series1_t0 = series1[:-1]
series1_t1 = series1[1:]

cc1 = np.corrcoef(series1_t0, series1_t1)


dseries1 = np.diff(series1_t1)
dseries1_t0 = dseries1[:-1]
dseries1_t1 = dseries1[1:]

cc2 = np.corrcoef(dseries1_t0, dseries1_t1)


r1, p1, nn, nb = pearsonr_ci(series1_t0, series1_t1, alpha=0.05)
r2, p2, vd, dv = pearsonr_ci(dseries1_t0, dseries1_t1, alpha=0.05)
