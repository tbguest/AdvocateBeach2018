'''
Tristan B Guest
8 Mar 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

%matplotlib qt5


homechar = "C:\\"

tide = "tide19"
position = "position2"
# vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2
# vidspec = "vid_1540311466" # pos4
colour = "yellow"
imgnum = "img010768.jpg" #p2
# imgnum = "img014226.jpg" #p3
# imgnum = "img001796.jpg" # p4

imgfile = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                           "images", "fromVideo", tide,position,vidspec,imgnum)
im = plt.imread(imgfile)


# load traj data
cobbledir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                           "cobble_tracking", tide, position, vidspec)

cobfiles = sorted(glob.glob(os.path.join(cobbledir, colour + '*.npy')))

posixtime0 = float(vidspec[-10:])

t_cobble = {}
pixloc = {}

fig, ax = plt.subplots(nrows=1,ncols=1, num="trajectory overlay")
ax.imshow(im)

rescale_pix_x = 1640/1000
rescale_pix_y = 1232/1000

for stone in sorted(cobfiles):
    # jnk = np.load(stone).item()
    # count = jnk['count']
    # t_cobble[int(stone[-6:-4])] = utime2yearday(posixtime0) + 3*60*60/86400 + (np.array(count) + startframe)/4/86400
    # # t_cobble[int(stone[-6:-4])] = utime2yearday(posixtime0) + (np.array(count) + 6801.0)/4/86400
    # pixloc[int(stone[-6:-4])] = jnk['position']



    jnk = np.load(stone).item()
    xpix = np.array(jnk['position'])[:,0] * rescale_pix_x
    ypix = np.array(jnk['position'])[:,1] * rescale_pix_x

    ax.plot(xpix, ypix,Linewidth=2)
    ax.plot(xpix[0], ypix[0], 'bo')
    ax.plot(xpix[-1], ypix[-1], 'ro')