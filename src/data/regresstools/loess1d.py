# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:07:41 2019

@author: Tristan Guest
"""

import numpy as np
from scipy import linalg

def loess(x, y, order=1, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    
    n = len(x)
    r = int(np.ceil(f * n))
    
    # number of relevant points?
    # h varies with distance from the walls, so the fitting interval is always
    # the same width. (rather than, eg, half the width at the borders)
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    
    # normalize, clip values between 0 and 1
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    
    # tricube weighting function
    w = (1 - w ** 3) ** 3
    
    yest = np.zeros(n)
    delta = np.ones(n)
    
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            
            # poly order
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    
#    # Identify number of points in weighting region
#    Jpts = r
#  
#    # sum weights for mear square interpolation error analysis
#    wi = np.sum(w**2)/Jpts**2
#    dif_wi = np.sum(((d - zi(j))*w)**2)/Jpts**2
#  
#    # compute mean square interpolation error
#    # sampling error:
#    ep(j) = wi
#  
#    # weighted mean square residual
#    qh(j) = 1/ep(j)*dif_wi
#  
#    # mean square interpolation error
##    sh(j) = ep(j)./(1 - ep(j)).*qh(j)
  
    return yest
