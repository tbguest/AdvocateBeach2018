'''
Tristan B Guest
8 Mar 2019
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os

%matplotlib qt5

# homechar = "C:\\"
homechar = os.path.expanduser('~')

tide = "tide19"
position = "position2"
# vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2
# vidspec = "vid_1540307860" # pos3
# vidspec = "vid_1540311466" # pos4
colour = "yellow"
# imgnum = "img001796.jpg" # p1
imgnum = "img010768.jpg" #p2
# imgnum = "img014226.jpg" #p3
# imgnum = "img003284.jpg" # p4

cmap = cm.tab10

# imgfile = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
#                            "images", "fromVideo", tide,position,vidspec,imgnum)
imgfile = os.path.join('/media', 'tristan2','Advocate2018_backup2', "data", "interim", \
                           "images", "fromVideo", tide,position,vidspec,imgnum)

splitdn = os.path.join(homechar,'Projects','AdvocateBeach2018','data','processed','images',\
        'coarsefine_split',tide,position)
splitfn = os.listdir(splitdn)[0]

splitimg = np.load(os.path.join(splitdn,splitfn), allow_pickle=True)

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
ax.imshow(splitimg, alpha=0.4)

# ax.plot(925, 436, 'wo')
# #841
# # 413
# # -1.904219368567567
# # 925
# # 436

rescale_pix_x = 1640/1000
rescale_pix_y = 1232/1000

counter = -1

for stone in sorted(cobfiles):
    counter += 1
    # jnk = np.load(stone).item()
    # count = jnk['count']
    # t_cobble[int(stone[-6:-4])] = utime2yearday(posixtime0) + 3*60*60/86400 + (np.array(count) + startframe)/4/86400
    # # t_cobble[int(stone[-6:-4])] = utime2yearday(posixtime0) + (np.array(count) + 6801.0)/4/86400
    # pixloc[int(stone[-6:-4])] = jnk['position']


# cmap = 'tab10'
    jnk = np.load(stone, allow_pickle=True).item()
    xpix = np.array(jnk['position'])[:,0] * rescale_pix_x
    ypix = np.array(jnk['position'])[:,1] * rescale_pix_x


    ax.plot(xpix, ypix,Linewidth=2, c='C3')
    ax.plot(xpix[0], ypix[0], 'o', markersize=8, markerfacecolor='r',markeredgewidth=2, markeredgecolor='k')
    ax.plot(xpix[-1], ypix[-1], '>', markersize=10, markerfacecolor='yellow',markeredgewidth=2, markeredgecolor='k')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    fig.tight_layout

saveFlag = 0
if saveFlag == 1:

    savedn = os.path.join(homechar,'Projects','AdvocateBeach2018','reports','figures','cobble_transport',tide)
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig.savefig(os.path.join(savedn, 'swash_trajectories_over_image_' + position + '.png'), dpi=1000, transparent=True)
    fig.savefig(os.path.join(savedn, 'swash_trajectories_over_image_' + position + '.pdf'), dpi=None, transparent=True)
