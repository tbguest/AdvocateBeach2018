import numpy as np
import pandas as pd
#import pylab as plt
import matplotlib.pyplot as plt

fn1 = "sonar71_2018-10-20-13_08.dat"
fn2 = "sonar72_2018-10-20-13_08.dat"
fn3 = "sonar73_2018-10-20-13_08.dat"
fn4 = "sonar74_2018-10-20-13_08.dat"
dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\20_10_2018\\AM\\"

with open(dn+fn1, 'rb') as f:
    clean_lines = ( line.replace(b'R',b'').replace(b'Oct',b'10').replace(b':',b' ') for line in f )
    range1 = np.genfromtxt(clean_lines,usecols=(5,),delimiter=' ')
    
with open(dn+fn2, 'rb') as f:
    clean_lines = ( line.replace(b'R',b'').replace(b'Oct',b'10').replace(b':',b' ') for line in f )
    range2 = np.genfromtxt(clean_lines,usecols=(5,),delimiter=' ')
    
with open(dn+fn3, 'rb') as f:
    clean_lines = ( line.replace(b'R',b'').replace(b'Oct',b'10').replace(b':',b' ') for line in f )
    range3 = np.genfromtxt(clean_lines,usecols=(5,),delimiter=' ')
    
with open(dn+fn4, 'rb') as f:
    clean_lines = ( line.replace(b'R',b'').replace(b'Oct',b'10').replace(b':',b' ') for line in f )
    range4 = np.genfromtxt(clean_lines,usecols=(5,),delimiter=' ') 
    
    
plt.figure
plt.plot(range1)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([500, 1500])
plt.show()    