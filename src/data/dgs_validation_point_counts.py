
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv

%matplotlib qt5

homechar = os.path.expanduser("~")

# sample date and location
date_str0 = ["Oct21", "Oct21", "Oct21", "Oct21", "Oct21", \
             "Oct21", "Oct25", "Oct25", "Oct25", "Oct25"]
tide0 = ["_bay1", "_bay2", "_bay3", "_horn1", "_horn2", "_horn3", \
         "_bay1", "_bay2", "_horn1", "_horn2"]

# for outdoor validation
lookupdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "references", \
                                   "img2sample_lookup_table_outdoorvalidation.csv")

imgnames = []
dates = []
locs = []
depths = []
iters = []

with open(lookupdir, 'rt') as f:
    lookup = csv.reader(f, delimiter=',')

    for row in lookup:
        imgnames.append(row[0])
        dates.append(row[1])
        locs.append(row[2])
        depths.append(row[3])
        iters.append(row[4])

subimg_xdiam = 200
subimg_ydiam = 200


mm_per_pixel = 0.13

for n in range(len(tide0)):

    particle_edges = []


# n=2
    # imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "raw", \
                      # "images", "OutdoorValidation", date_str0[n] + tide0[n])
    imgdir = os.path.join("/media","tristan2","Advocate2018_backup2", "data", "raw", \
                      "images", "OutdoorValidation", date_str0[n] + tide0[n])

    allfiles = sorted(glob.glob(os.path.join(imgdir,'*cropped.jpg')))

    dep_ind = -1
    im = plt.imread(allfiles[dep_ind])

    ydim, xdim = np.shape(im)[:2]
    dy = ydim/10
    dx = xdim/10

    x = np.arange(0,xdim, dx)[1:]
    y = np.arange(0,ydim, dy)[1:]

    xx, yy = np.meshgrid(x, y)

    plt.figure(5)
    plt.imshow(im)
    plt.plot(xx,yy,'w.')

    # xxlin = np.reshape(xx, len(xx)*len(yy))
    # yylin = np.reshape(yy, len(xx)*len(yy))

    # counter = -1

    for xp in x:
        for yp in y:
# xp=x[0]
# yp=y[0]

            # counter += 1

            # make sure sub image boundaries are within image lims
            if int(xp)-subimg_xdiam < 0:
                xrange = np.arange(0,int(xp)+subimg_xdiam)
            elif int(xp)+subimg_xdiam > xdim:
                xrange = np.arange(int(xp)-subimg_xdiam,xdim)
            else:
                xrange = np.arange(int(xp)-subimg_xdiam,int(xp)+subimg_xdiam)

            if int(yp)-subimg_ydiam < 0:
                yrange = np.arange(0,int(yp)+subimg_ydiam)
            elif int(yp)+subimg_ydiam > ydim:
                yrange = np.arange(int(yp)-subimg_ydiam,ydim)
            else:
                yrange = np.arange(int(yp)-subimg_ydiam,int(yp)+subimg_ydiam)

            subimg = im[np.min(yrange):np.max(yrange), np.min(xrange):np.max(xrange)]


            # plt.figure(2)
            # plt.clf()
            # plt.imshow(subimg)
            # plt.plot(xp % len(subimg[:,0]),yp % len(subimg[0,:]),'+')
            # plt.draw()

            plt.figure(2)
            plt.clf()
            plt.imshow(im)
            # plt.gca().invert_yaxis()
            plt.xlim(np.min(xrange),np.max(xrange))
            plt.ylim(np.max(yrange),np.min(yrange))
            plt.plot(xp,yp,'+')
            plt.draw()

            pts = plt.ginput(2)

            xdiff = np.abs(pts[1][0] - pts[0][0])*mm_per_pixel
            particle_edges.append(xdiff)


    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','data','processed',\
                'grainsize','validation','point-counts')

    # find img index for matching with sample #, depth, etc
    imgind0 = allfiles[dep_ind].split('/')[-1]
    imgind = str(int(imgind0.split('-')[0][4:]))
    Idepth = imgnames.index(imgind)

    np.save(os.path.join(savedn, date_str0[n] + tide0[n] + '_' + depths[Idepth] + '.npy'), particle_edges)


plt.figure(88)
plt.hist(particle_edges)
