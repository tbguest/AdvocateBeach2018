
# coding: utf-8

# # 16 Oct 2018 - Initial exploration of range sensor data

# In[16]:


import numpy as np
import pandas as pd
#import pylab as plt
import matplotlib.pyplot as plt


# In[9]:


fn1 = "rng10152018_pi71.001"
fn2 = "rng10152018_pi72.001"
fn3 = "rng10152018_pi73.001"
fn4 = "rng10152018_pi74.001"
dn = "C:\\Projects\\AdvocateBeach2018\\data\\raw\\range_data\\15_10_2018\\PM\\"

data1 = np.genfromtxt(dn+fn1,delimiter='R',usecols=(1,))
data2 = np.genfromtxt(dn+fn2,delimiter='R',usecols=(1,))
data3 = np.genfromtxt(dn+fn3,delimiter='R',usecols=(1,))
data4 = np.genfromtxt(dn+fn4,delimiter='R',usecols=(1,))


# In[10]:


plt.figure
plt.plot(data1)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([500, 1500])
plt.show()


# Note the delayed returns (e.g. bed farther away than sould be) -- apparently following or preceding runup events. Air bubbles in swash?

# In[15]:


plt.figure
plt.plot(data2)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([500, 1500])
plt.show()


# In[12]:


plt.figure
plt.plot(data3)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([500, 1500])
plt.show()


# NB: change in bed elevation at ~5000 is a clump of seaweed. 

# In[14]:


plt.figure
plt.plot(data4)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([500, 1500])
plt.show()


# Try a moving average filter (should really clean erroneous values first):

# In[18]:


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N


# In[22]:


N = 10
data1_rm = running_mean(data1, N) 


# In[23]:


plt.figure
plt.plot(data1)
plt.plot(data1_rm)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([800, 1200])
plt.show()


# In[27]:


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


# In[35]:


import numpy
data_sm = smooth(data,window_len=257,window='hanning') #x,window_len=11,window='hanning'


# In[36]:


plt.figure
plt.plot(data1)
plt.plot(data_sm)
plt.xlabel('index')
plt.ylabel('Range [mm]')
plt.ylim([800, 1200])
plt.show()

