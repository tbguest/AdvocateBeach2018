# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:52:42 2018

@author: Tristan Guest
"""

import os
import numpy as np
import csv

# homechar = "C:\\"
homechar = os.path.expanduser('~')
fn = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', 'raw', 'grain_size', 'cusp_samples.csv')
outdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', 'interim', 'grainsize', 'sieved')


#bins = [45.0, 31.5, 22.4, 16.0, 8.0, 6.7, 4.0, 2.8, 2.0, 1.4, 1.0, 0.707, 0.5, 0.0]
#bins = [22.4, 16.0, 8.0, 6.7, 4.0, 2.8, 2.0, 1.4, 1.0, 0.707, 0.5, 0.25]
#bins = [31.5, 22.4, 16.0, 8.0, 6.7, 4.0, 2.8, 2.0, 1.4, 1.0, 0.707, 0.5]

bins0 = [63.0, 45.0, 31.5, 22.4, 16.0, 8.0, 6.7, 4.0, 2.8, 2.0, 1.4, 1.0]
bins = np.flipud(np.diff(np.flipud(bins0))/2 + np.flipud(bins0[1:]))

with open(fn, 'rt') as csvfile:

    data = np.genfromtxt(csvfile, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), \
                         delimiter=',', skip_header=1, dtype=None)

    for row in data:
        foo = np.array(row).item()
#        gs = foo[2:-1]
        gs = foo[2:-4]

        date = foo[0].decode("utf-8")
        sampID = foo[1].decode("utf-8")

        location, depth = sampID.split('-')

        total = np.sum(gs)
        gsfreq = gs/total

        cumsum = np.cumsum(gsfreq)

        sample = {'date': date, 'location': location, 'depth': depth[:-2], \
                  'grain size bins': bins, 'grain size frequencies': gsfreq, \
                  'cumulative sum': cumsum}

        fname = os.path.join(outdir, 'Oct' + date[0:2] + '_' + location + '_' + depth[:-2] + '.npy')
        np.save(fname, sample)
