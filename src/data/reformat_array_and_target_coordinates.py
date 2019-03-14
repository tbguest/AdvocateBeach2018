'''
Tristan B Guest
6 Mar 2019
'''

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sys


def rotate_coordainates(nrth, east, elvn):

    ### TEMP:
    # fname = posfiles[0]
    ###

    # UTM coords of origin (Adv2015)
    originx = 3.579093296000000e+05;
    originy = 5.022719408400000e+06;

    northing = nrth - originx
    easting = east - originy
    z = elvn

    # rotation angle from 2015 stake coordinates
    theta_rot = np.pi + np.arctan((357899.3979000000 - 357908.4006000000)/(5022717.801400000 - 5022709.436700001))

    # rotation matrix
    R = np.array([np.cos(theta_rot), -np.sin(theta_rot), np.sin(theta_rot), np.cos(theta_rot)]).reshape((2, 2))

    # apply to coordinates
    Ymat = np.array([northing, easting])
    Ymat_p = np.matmul(R,Ymat)
    y = Ymat_p[0]
    x = Ymat_p[1]

    return x, y, z


def use_endpoints(data):

    north0 = data[:,1]
    east0 = data[:,2]
    z0 =  data[:,3]

    nrth = np.linspace(north0[0], north0[1], 4)
    east = np.linspace(east0[0], east0[1], 4)
    elvn = np.linspace(z0[0], z0[1], 4)

    return nrth, east, elvn



# Change these: #
tide = '15'
# chunk = '1' # for array positions
# position = '1' # for cobble cam positions
# date = '21_10_2018'
date = '21_10_2018'

homechar = "C:\\"

##### INPUT #####
# for locations of each pi housing in array (WITHOUT POST)
array_dir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'raw', 'GPS', 'pi_locations', date, 'pi_array')
# for GCPs in cobble cam data
target_dir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'raw', 'GPS', 'GCPs', date, 'cobble_cam')

##### OUTPUT #####
# save array location data in npy format
save_array_dir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'GPS', 'array', 'tide' + tide)
# save GCP target data in npy format
save_target_dir = os.path.join(homechar, 'Projects', 'AdvocateBeach2018', 'data', \
                           'processed', 'GPS', 'cobble_cam_targets', 'tide' + tide)



# main:
posfiles = sorted(glob.glob(os.path.join(array_dir, 'position*')))
numfiles = len(posfiles)

counter = 0

for pos in posfiles:

    counter += 1

    with open(pos, 'rb') as f:
        clean_lines = ( line.replace(b'STK',b'').replace(b' ',b',') for line in f )
        data = np.genfromtxt(clean_lines,usecols=(0,1,2,3,4,),delimiter=',')

    if len(data) < 4:
        nrth, east, elvn = use_endpoints(data)
    else:
        nrth = data[:,1]
        east = data[:,2]
        elvn = data[:,3]

    x, y, z = rotate_coordainates(nrth, east, elvn)

    fig, (ax1, ax2) = plt.subplots(2,1,num=str(counter))
    ax1.plot(x, y, 'o')
    ax1.set_ylabel('y [m]')
    ax1.set_xlabel('x [m]')
    ax2.plot(x, z, 'o')
    ax2.set_ylabel('z [m]')
    ax2.set_xlabel('x [m]')


    # reorder to verify proper order wrt pis
    Ipis = np.argsort(x)
    xsort = x[Ipis]
    ysort = y[Ipis]
    zsort = z[Ipis]

    coords = {'x': xsort, 'y': ysort, 'z': zsort}

    if not os.path.exists(save_array_dir):
    	try:
    		os.makedirs(save_array_dir)
    	except OSError as exc: # Guard against race condition
    		if exc.errno != errno.EEXIST:
    			raise

    # save as npy for quick access
    np.save(os.path.join(save_array_dir, 'array_position' + pos[-5] + '.npy'), coords)
    # np.save(os.path.join(save_target_dir, 'cobblecam_position' + position + '.npy'), ...)
