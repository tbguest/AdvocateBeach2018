#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process survey grid data: organize into tide-delimited folders, with 
longshore, cross-shore, and dense grids separated"""

import numpy as np
import math
import os


# for portability
homechar = "C:\\"

dn = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                  "GPS", "raw_by_tide")

dnout = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                  "GPS", "by_tide")

# All files:

# fn = ["15_10_2018_A.txt", "16_10_2018_A.txt", ...
fn = ["17_10_2018_A.txt", "18_10_2018_A.txt", "19_10_2018_A.txt", "20_10_2018_A.txt", \
    "21_10_2018_A.txt", "21_10_2018_B.txt", "22_10_2018_A.txt", "22_10_2018_B.txt", "23_10_2018_A.txt", \
    "23_10_2018_B.txt", "24_10_2018_A.txt", "24_10_2018_B.txt", "25_10_2018_A.txt", \
    "25_10_2018_B.txt", "26_10_2018_A.txt", "26_10_2018_B.txt", "27_10_2018_A.txt", \
    "27_10_2018_B.txt", "21_10_2018_B_longshore1.txt", "22_10_2018_A_longshore1.txt", \
    "22_10_2018_B_longshore1.txt", "23_10_2018_A_longshore1.txt", \
    "23_10_2018_B_longshore1.txt", "24_10_2018_A_longshore1.txt", \
    "24_10_2018_B_longshore1.txt", "25_10_2018_A_longshore1.txt", \
    "25_10_2018_B_longshore1.txt", "26_10_2018_A_longshore1.txt", \
    "26_10_2018_B_longshore1.txt", "27_10_2018_A_longshore1.txt", \
    "27_10_2018_B_longshore1.txt"]

# tide number - 1 being first of experiment (Sunday AM UTC)
#{"15_10_2018_A.txt": 3, "16_10_2018_A.txt": 5, \
tide = {"17_10_2018_A.txt": 7, "18_10_2018_A.txt": 9, "19_10_2018_A.txt": 11, "20_10_2018_A.txt": 13, \
    "21_10_2018_A.txt": 14, "21_10_2018_B.txt": 15, "22_10_2018_A.txt": 16, "22_10_2018_B.txt": 17, "23_10_2018_A.txt": 18, \
    "23_10_2018_B.txt": 19, "24_10_2018_A.txt": 20, "24_10_2018_B.txt": 21, "25_10_2018_A.txt": 22, \
    "25_10_2018_B.txt": 23, "26_10_2018_A.txt": 24, "26_10_2018_B.txt": 25, "27_10_2018_A.txt": 26, \
    "27_10_2018_B.txt": 27, "21_10_2018_B_longshore1.txt": 15, "22_10_2018_A_longshore1.txt": 16, \
    "22_10_2018_B_longshore1.txt": 17, "23_10_2018_A_longshore1.txt": 18, \
    "23_10_2018_B_longshore1.txt": 19, "24_10_2018_A_longshore1.txt": 20, \
    "24_10_2018_B_longshore1.txt": 21, "25_10_2018_A_longshore1.txt": 22, \
    "25_10_2018_B_longshore1.txt": 23, "26_10_2018_A_longshore1.txt": 24, \
    "26_10_2018_B_longshore1.txt": 25, "27_10_2018_A_longshore1.txt": 26, \
    "27_10_2018_B_longshore1.txt": 27}


# with respective grids:

# grid = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
grid = [2, 2, 2, 2, \
        3, 3, 3, 3, 3, \
        3, 3, 3, 3, \
        3, 3, 3, 3, \
        3, 4, 4, \
        4, 4, \
        4, 4, \
        4, 4, \
        4, 4, \
        4, 4, \
        4]


def import_RTK(fname, gridnum):

    with open(fname, 'rb') as f:
        clean_lines = ( line.replace(b'STK',b'').replace(b' ',b',') for line in f )
        data = np.genfromtxt(clean_lines,usecols=(0,1,2,3,4,),delimiter=',')

    # UTM coords of origin (Adv2015)
    originx = 3.579093296000000e+05;
    originy = 5.022719408400000e+06;

    northing = data[:,1] - originx
    easting = data[:,2] - originy
    elevation = data[:,3]
    stkID = data[:,4]

    # rotation angle from 2015 stake coordinates
    theta_rot = math.pi + math.atan((357899.3979000000 - 357908.4006000000)/(5022717.801400000 - 5022709.436700001))
    
    # rotation matrix
    R = np.array([math.cos(theta_rot), -math.sin(theta_rot), math.sin(theta_rot), math.cos(theta_rot)]).reshape((2, 2))

    # apply to coordinates
    Ymat = np.array([northing, easting])
    Ymat_p = np.matmul(R,Ymat)
    y = Ymat_p[0]
    x = Ymat_p[1]

    # indices for each grid type
    Izl = []
    ILl1 = []
    ILl2 = []
    Ida1 = []
    Ida2 = []

    if gridnum == 1:
        zlineID = list(range(70,101))
        LlineID = list(range(284,360))

        # took me a while to figure out this indexing -- hang on to this
        for stk in stkID:
            if stk in zlineID:
                Izl.append(list(np.where(stkID==stk))[0][0])
            if stk in LlineID:
                ILl1.append(list(np.where(stkID==stk))[0][0])

    elif gridnum == 2:

        zlineID = list(range(145,175))
        LlineID = list(range(176,202))
        denseID = list(range(0,145))

        for stk in stkID:
            if stk in zlineID:
                Izl.append(list(np.where(stkID==stk))[0][0])
            if stk in LlineID:
                ILl1.append(list(np.where(stkID==stk))[0][0])
            if stk in denseID:
                Ida1.append(list(np.where(stkID==stk))[0][0])
    
    elif gridnum == 3:

        zlineID = list(range(145,175))
        LlineID = list(range(176,202))
        denseID = list(range(0,145))

        for stk in stkID:
            if stk in zlineID:
                Izl.append(list(np.where(stkID==stk))[0][0])
            if stk in LlineID:
                ILl2.append(list(np.where(stkID==stk))[0][0])
            if stk in denseID:
                Ida2.append(list(np.where(stkID==stk))[0][0])

    elif gridnum == 4:

        LlineID = list(range(176,202))

        for stk in stkID:
            if stk in LlineID:
                ILl1.append(list(np.where(stkID==stk))[0][0])
                

    zline = (y[Izl], x[Izl], elevation[Izl])
    longshore1 = (y[ILl1], x[ILl1], elevation[ILl1])
    longshore2 = (y[ILl2], x[ILl2], elevation[ILl2])
    dense_array1 = (y[Ida1], x[Ida1], elevation[Ida1])
    dense_array2 = (y[Ida2], x[Ida2], elevation[Ida2])
    
    zline = {"y": y[Izl], "x": x[Izl], \
             "z": elevation[Izl]}
    longshore1 = {"y": y[ILl1], "x": x[ILl1], \
                  "z": elevation[ILl1]}
    longshore2 = {"y": y[ILl2], "x": x[ILl2], \
                  "z": elevation[ILl2]}
    dense_array1 = {"y": y[Ida1], "x": x[Ida1], \
                    "z": elevation[Ida1]}
    dense_array2 = {"y": y[Ida2], "x": x[Ida2], \
                    "z": elevation[Ida2]}
    
#    coords = {"zline": zline,
#              "longshore1": longshore1,
#              "longshore2": longshore2,
#              "dense_array1": dense_array1
#              "dense_array2": dense_array2}
    
    # output
    if "longshore1" in fname:
        fout = fname[-27:-15]
    else:
        fout = fname[-16:-4]
        
    tidenum = str(tide[fout + ".txt"])    
    
#    svdir = os.path.join(dnout, fout)
    svdir = os.path.join(dnout, "tide" + tidenum)
        
    if not os.path.isdir(svdir):
        os.mkdir(svdir)
        
#    fnout = os.path.join(svdir, "zline.npy")     
    
    # don't save over existing writes
    if "longshore1" in fname:    
        np.save(os.path.join(svdir, "longshore1.npy") , longshore1)
    else:
        np.save(os.path.join(svdir, "zline.npy") , zline)
        np.save(os.path.join(svdir, "longshore1.npy") , longshore1)
        np.save(os.path.join(svdir, "longshore2.npy") , longshore2)
        np.save(os.path.join(svdir, "dense_array1.npy") , dense_array1)
        np.save(os.path.join(svdir, "dense_array2.npy") , dense_array2)
    
#    return zline, longshore, dense_array


# MAIN LOOP
def main():   
    
    for m in range(0, len(fn)): 
            
        fname = os.path.join(dn, fn[m])
    
        import_RTK(fname, grid[m])
        
    #    # NB: coords[zl,Ll,da][x,y,z]    
main()    
