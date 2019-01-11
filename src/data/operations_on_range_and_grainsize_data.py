# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:45:01 2019

@author: Owner

plot range and grain size data, by chunk
"""

import numpy as np 
import os
from glob import glob
import matplotlib.pyplot as plt
from regresstools import lowess


tide = '19'
chunk = '3'

homechar = "C:\\"      

gsdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'grainsize', 'pi_array', 'tide' + tide)
    
rngdir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'range_data', 'bed_level', 'tide' + tide)


aaa = np.load(os.path.join(gsdir, 'chunk' + chunk + '.npy')).item()
bbb = np.load(os.path.join(rngdir, 'chunk' + chunk + '.npy')).item()

f0 = 0.25
iters = 1

mgs_fit1 = lowess.lowess(aaa['pi71']['tvec'], aaa['pi71']['mgs'], f=f0, iter=iters)  
mgs_fit2 = lowess.lowess(aaa['pi72']['tvec'], aaa['pi72']['mgs'], f=f0, iter=iters)  
mgs_fit3 = lowess.lowess(aaa['pi73']['tvec'], aaa['pi73']['mgs'], f=f0, iter=iters)  
mgs_fit4 = lowess.lowess(aaa['pi74']['tvec'], aaa['pi74']['mgs'], f=f0, iter=iters)  

sort_fit1 = lowess.lowess(aaa['pi71']['tvec'], aaa['pi71']['sort'], f=f0, iter=iters)  
sort_fit2 = lowess.lowess(aaa['pi72']['tvec'], aaa['pi72']['sort'], f=f0, iter=iters)  
sort_fit3 = lowess.lowess(aaa['pi73']['tvec'], aaa['pi73']['sort'], f=f0, iter=iters)  
sort_fit4 = lowess.lowess(aaa['pi74']['tvec'], aaa['pi74']['sort'], f=f0, iter=iters)  


fig = plt.figure(1)

ax1 = fig.add_subplot(311)
ax1.plot(bbb['pi71']['tvec'], bbb['pi71']['range'], '.C0')
ax1.plot(bbb['pi72']['tvec'], bbb['pi72']['range'], '.C1')
ax1.plot(bbb['pi73']['tvec'], bbb['pi73']['range'], '.C2')
ax1.plot(bbb['pi74']['tvec'], bbb['pi74']['range'], '.C3')
ax1.autoscale(enable=True, axis='x', tight=True)

ax2 = fig.add_subplot(312)
ax2.plot(aaa['pi71']['tvec'], aaa['pi71']['mgs'], '.C0')
ax2.plot(aaa['pi72']['tvec'], aaa['pi72']['mgs'], '.C1')
ax2.plot(aaa['pi73']['tvec'], aaa['pi73']['mgs'], '.C2')
ax2.plot(aaa['pi74']['tvec'], aaa['pi74']['mgs'], '.C3')
ax2.plot(aaa['pi71']['tvec'], mgs_fit1, 'C0', Linewidth=2)
ax2.plot(aaa['pi72']['tvec'], mgs_fit2, 'C1', Linewidth=2)
ax2.plot(aaa['pi73']['tvec'], mgs_fit3, 'C2', Linewidth=2)
ax2.plot(aaa['pi74']['tvec'], mgs_fit4, 'C3', Linewidth=2)
ax2.autoscale(enable=True, axis='x', tight=True)

ax3 = fig.add_subplot(313)
ax3.plot(aaa['pi71']['tvec'], aaa['pi71']['sort'], '.C0')
ax3.plot(aaa['pi72']['tvec'], aaa['pi72']['sort'], '.C1')
ax3.plot(aaa['pi73']['tvec'], aaa['pi73']['sort'], '.C2')
ax3.plot(aaa['pi74']['tvec'], aaa['pi74']['sort'], '.C3')
ax3.plot(aaa['pi71']['tvec'], sort_fit1, 'C0', Linewidth=2)
ax3.plot(aaa['pi72']['tvec'], sort_fit2, 'C1', Linewidth=2)
ax3.plot(aaa['pi73']['tvec'], sort_fit3, 'C2', Linewidth=2)
ax3.plot(aaa['pi74']['tvec'], sort_fit4, 'C3', Linewidth=2)
ax3.autoscale(enable=True, axis='x', tight=True)


