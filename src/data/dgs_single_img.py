from dgs import *
import glob
import os
import errno
import numpy as np

# fn = "/media/tristan/Advocate2018_backup2/data/processed/grainsize/pi_array/tide19/reprocessed_x08_dynamicmaxscale/chunk1.npy"

# gs = np.load(fn, allow_pickle=True)
# gs["mgs"]

# # image_file = "/home/tristan/Documents/manuscripts/msd-swash/jmse-manuscript-2021/figures/review/tide27-pi71_img1540662307-549075.jpg"


# image_file = "/home/tristan/Documents/manuscripts/msd-swash/jmse-manuscript-2021/figures/review/img1540661200-045598-cropped.jpg"
image_file = "/home/tristan/Documents/manuscripts/msd-swash/jmse-manuscript-2021/figures/review/img1540662490-996864-cropped.jpg"


# change camera height on the 18th (tide 9)
resolution = 0.13
density = 10  # process every 10 lines
dofilter = 1  # filter the imagei
notes = 8  # notes per octave
maxscale = 8  # Max scale as inverse fraction of data length
verbose = 1  # print stuff to screen
x = 0.8
# dgs_stats = dgs(
#     image_file, density, resolution, dofilter, maxscale, notes, verbose, x
# )
data_out = dgs(image_file, resolution, maxscale, verbose, x)
