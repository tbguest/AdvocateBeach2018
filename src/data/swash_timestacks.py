
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import cv2 as cv
%matplotlib qt5
import imutils
import skimage.morphology


def tstack2binmask(tstack, change_thresh):
# produces a binary mask of high/low pixel intensity change from a timestack images
# change_thresh: abs variation threshold in pix intensity between timesteps  [20]

    # binarize difference matrix
    diffpix = np.diff(tstack)
    bin_diffpix = np.zeros((1640, len(imgs)-1))
    bin_diffpix[np.abs(diffpix) > change_thresh] = 1

    # kernel = np.ones((3,3),np.uint8)
    # jnk = cv.erode(bin_diffpix.astype(float),kernel,iterations = 1)
    # dpix_open = dilation = cv.dilate(jnk,kernel,iterations = 1)
    # dpix_close = cv.erode(cv.dilate(bin_diffpix.astype(float),kernel,iterations = 1),kernel,iterations = 3)

    # morphological closing and opening, to fill swash region
    se1 = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    se2 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    foo = cv.morphologyEx(bin_diffpix, cv.MORPH_CLOSE, se1)
    clsopen = cv.morphologyEx(foo, cv.MORPH_OPEN, se2)

    # clean noise
    remv_islands = skimage.morphology.remove_small_objects(clsopen.astype(bool), min_size=4096, connectivity=1, in_place=False)

    return remv_islands



def get_timeseries(bin_mask, thresh):
# returns time series of swash front
# inputs
# bin_mask: binary mask of high(swash)/low(beach) intensity change
# thresh: number of high change pixels to count before accepting as shoreline

    front = np.zeros(len(bin_mask[1,:]))

    # cycle through colums (timesteps)
    for j in range(len(bin_mask[1,:])):
        col = bin_mask[:,j]

        # reverse column so 0 is offshore
        reversed_col = col[::-1]

        counter = 0
        # loop over pixels
        for k in range(len(reversed_col)):
            # add to counter if region of high change
            if reversed_col[k] == True:
                counter += 1
            else:
                counter = 0

            # if count has been reached, identify swash edge
            # pixcount is threshold count of high change values, to pass over noise
            pixcount = 40
            if counter > pixcount:
                # front.append(k - pixcount)
                front[j] = len(reversed_col) - k + pixcount
                break

    return front

homechar = "C:\\"

tide = "tide19"
position = "position1"
# position = "position3"
# position = "position4"
vidspec = "vid_1540304255" # pos1
# vidspec = "vid_1540307860" # pos2
# vidspec = "vid_1540311466" # pos4


imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                      "images", "fromVideo", tide, position, vidspec)

imgs = sorted(glob.glob(os.path.join(imgdir, 'img*.jpg')))

tstack0200 = np.zeros((1640, len(imgs)))
tstack0400 = np.zeros((1640, len(imgs)))
tstack0600 = np.zeros((1640, len(imgs)))
tstack0800 = np.zeros((1640, len(imgs)))
tstack1000 = np.zeros((1640, len(imgs)))

for ii in range(len(imgs)):

    im = plt.imread(imgs[ii])

    # plt.figure(1)
    # im = plt.imread(im)
    # plt.imshow(im)

    # r, g, b = cv.split(im)
    # is_blu = cv.inRange(b, 150, 255)
    # is_grn = cv.inRange(g, 150, 255)
    # is_red = cv.inRange(r, 150, 255)
    # wht_mask = np.logical_and(is_red, is_grn)
    # wht_mask = np.logical_and(wht_mask, is_blu)

    gray_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # line0200 = gray_image[200,]
    # line0400 = gray_image[400,]
    # line0600 = gray_image[600,]
    # line0800 = gray_image[800,]
    # line1000 = gray_image[1000,]

    tstack0200[:,ii] = gray_image[200,]
    # tstack0400[:,ii] = gray_image[400,]
    tstack0600[:,ii] = gray_image[600,]
    # tstack0800[:,ii] = gray_image[800,]
    tstack1000[:,ii] = gray_image[1000,]

    # tstack0200[:,ii] = wht_mask[200,]
    # tstack0400[:,ii] = wht_mask[400,]
    # tstack0600[:,ii] = wht_mask[600,]
    # tstack0800[:,ii] = wht_mask[800,]
    # tstack1000[:,ii] = wht_mask[1000,]


remv_islands0200 = tstack2binmask(tstack0200, 20)
front0200 = get_timeseries(remv_islands0200, 30)

remv_islands0600 = tstack2binmask(tstack0600, 20)
front0600 = get_timeseries(remv_islands0600, 30)

remv_islands1000 = tstack2binmask(tstack1000, 20)
front1000 = get_timeseries(remv_islands1000, 30)

# use vidspec and frame nums to populate time vector
t_video = float(vidspec[4:])
t_firstframe = float(imgs[0][-10:-4])
t_lastframe = float(imgs[-1][-10:-4])
tvec = t_video + np.linspace(t_firstframe/4, t_lastframe/4, len(imgs))

timeseries = {'timevec': tvec[1:], 'line0200': front0200, 'line0600': front0600, 'line1000': front1000}
timestacks = {'timevec': tvec[1:], 'line0200': tstack0200, 'line0600': tstack0600, 'line1000': tstack1000}


# save
savedir = os.path.join('C:\\', 'Projects', 'AdvocateBeach2018', 'data', 'interim', 'swash', tide, position)
dn1 = os.path.join(savedir, 'timeseries', vidspec)
dn2 = os.path.join(savedir, 'timestacks', vidspec)

if not os.path.exists(dn1):
    try:
        os.makedirs(dn1)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

if not os.path.exists(dn2):
    try:
        os.makedirs(dn2)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

np.save(os.path.join(dn1, 'timeseries.npy'), timeseries)
np.save(os.path.join(dn2, 'timestacks.npy'), timestacks)



# plt.figure(10, figsize=(18, 16))
# plt.imshow(diffpix)
# plt.tight_layout()
# plt.colorbar()


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 16))
ax1.imshow(tstack0200,origin='lower')
ax1.plot(np.array(front0200), 'r')
ax1.set_aspect('auto')
ax2.imshow(remv_islands0200,origin='lower')
ax2.plot(np.array(front0200), 'r')
ax2.set_aspect('auto')

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 16))
ax1.imshow(tstack0600,origin='lower')
ax1.plot(np.array(front0600), 'r')
ax1.set_aspect('auto')
ax2.imshow(remv_islands0600,origin='lower')
ax2.plot(np.array(front0600), 'r')
ax2.set_aspect('auto')

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 16))
ax1.imshow(tstack1000,origin='lower')
ax1.plot(np.array(front1000), 'r')
ax1.set_aspect('auto')
ax2.imshow(remv_islands1000,origin='lower')
ax2.plot(np.array(front1000), 'r')
ax2.set_aspect('auto')


# tvec[600]
# tvec[1500]

# tide19-pos4
tvec[800]
tvec[-1]

def utime2yearday(unixtime):

    from datetime import datetime
    import time

    dt = datetime(2018, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday

a1 = utime2yearday(tvec[3500] + 3*60*60)
a2 = utime2yearday(tvec[5500] + 3*60*60)
#
# plt.figure(2, figsize=(18, 16))
# # plt.imshow(tstack0600[:,500:800],origin='lower')
# plt.imshow(tstack0600,origin='lower')
# plt.plot(np.array(front), 'r')
# plt.axes().set_aspect('auto')


# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(18, 16))
# ax1.imshow(bin_diffpix)
# # ax2.imshow(dpix_close)
# ax2.imshow(clsopen)
# ax3.imshow(remv_islands)
# # plt.tight_layout()
