# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:49:57 2019

@author: Tristan Guest
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import errno
import cv2 as cv


homechar = "C:\\"

tide = "tide19"
position = "position2"
#vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2
colour = "orange"

savedir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                           "cobble_tracking", tide, position, vidspec, colour)

#C:\Projects\AdvocateBeach2018\data\interim\cobble_tracking\tide19\position1\vid_1540304255

imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                      "images", "fromVideo", tide, position, vidspec)

#C:\Projects\AdvocateBeach2018\data\interim\images\fromVideo\tide19\position2\vid_1540307860

imgs = sorted(glob.glob(os.path.join(imgdir, 'img*.jpg')))

# repositories of stone locations and trajectories
bank = []

# if returning to this, load most recent file to start with
predef_locs = np.sort(glob.glob(os.path.join(savedir, '*.npy')))

start_img = 0
dimg = 4 # use every 'dimg' frames for analysis [4 is default]

if len(predef_locs) > 0:
    last_file = predef_locs[-1][-13:-4]
    last_file_index = imgs.index(os.path.join(imgdir, last_file + '.jpg'))

    old_bank = np.load(predef_locs[-1]).item()['positions']
    for i in range(len(old_bank)):
        bank.append(tuple(old_bank[i]))

    start_img = last_file_index + dimg

# main loop

for file in imgs[start_img::dimg]:

    # bank of stone trajectories. This gets emptied for every new image.
    traj_bank = []

    im = plt.imread(file)
    # im = plt.imread(imgs[start_img + 304])


    plt.figure(1).clf()
    plt.imshow(im)
    plt.tight_layout()
    # plt.draw()
    # plt.show()

    np.disp("Image: " + str(file[-13:]))

    # Step 1
    if bank:
        plt.figure(1)
        plt.plot(np.array(bank)[:,0], np.array(bank)[:,1], 'ro')
        plt.draw()

        ############

        # Final step: are any of the stones no longer visible?
        np.disp("Click, then press enter if any stones are gone or no longer visible. Otherwise, press enter.")
        gonepts = plt.ginput(-1, timeout=0, show_clicks=True)

        if gonepts:

            new_bank = []

            # loop over points. Actually, this should be done first.
            for n in range(len(bank)):

                plt.figure(1).clf()
                plt.imshow(im)
    #                plt.tight_layout()
                plt.plot(np.array(bank)[:,0], np.array(bank)[:,1], 'ro')
                plt.plot(np.array(bank)[n,0], np.array(bank)[n,1], 'co')
                plt.draw()

                np.disp("If the cyan stone can no longer be seen, click anywhere, then hit enter. Otherwise, hit enter.")
                foopt = plt.ginput(-1, timeout=0, show_clicks=True)

                if not foopt:
                    new_bank.append(bank[n])

            bank = new_bank # assignment problem?

            plt.figure(1).clf()
            plt.imshow(im)
#                plt.tight_layout()
            if bank:
                plt.plot(np.array(bank)[:,0], np.array(bank)[:,1], 'ro')
            plt.draw()

        ############


    np.disp("Click on any new or moved stones.")
    newpts = np.array(plt.ginput(-1, timeout=0, show_clicks=True))

    if len(newpts) > 0:

        plt.plot(newpts[:,0], newpts[:,1], 'go')
        plt.draw()
        np.disp("Have any of the indicated stones moved? If so, click anywhere then press enter. If not, press enter.")
        jnkpts = plt.ginput(-1, timeout=0)

        # remove old points from the bank if they've moved
        if jnkpts:

            for j in range(len(newpts)):

                plt.figure(1).clf()
                plt.imshow(im)
#                plt.tight_layout()
                plt.plot(np.array(bank)[:,0], np.array(bank)[:,1], 'ro')
                plt.plot(newpts[:,0], newpts[:,1], 'go')
                plt.plot(newpts[j,0], newpts[j,1], 'bo')
                plt.draw()


                np.disp("If the blue point indicates a new stone, hit enter. If moved, click the old stone location.")
                movedpt = np.array(plt.ginput(-1, timeout=0, show_clicks=True))

                if len(movedpt) < 1:
                    continue

                else:
                    # find closest point in bank. save transport info and remove old location from bank
                    xjnk = np.array(bank)[:,0] - movedpt[0][0]
                    yjnk = np.array(bank)[:,1] - movedpt[0][1]
                    Imin = np.argmin(np.abs(xjnk**2 + yjnk**2))

                    traj_bank.append([np.array(bank[Imin]), newpts[j]])

                    del bank[Imin]





    # add new pts to bank
    [bank.append(newpts[i]) for i in range(len(newpts))]

    fn = file[-13:-4] + '.npy'

    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    timestamp = float(file[-24:-14]) + float(file[-10:-4])/4.0
    imgdata = {'timestamp': timestamp, 'positions': bank, 'trajectories': traj_bank}

    # np.save(os.path.join(savedir, fn), imgdata)

#    plt.figure(2).clf()
#    plt.plot(np.array(bank)[:,0], np.array(bank)[:,1], 'ko')
#    for p in range(len(traj_bank)):
#        plt.plot([np.array(traj_bank)[p][0][0], np.array(traj_bank)[p][1][0]], [np.array(traj_bank)[p][0][1], np.array(traj_bank)[p][1][1]])
