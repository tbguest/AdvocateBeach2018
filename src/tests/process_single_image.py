#!/usr/bin/env python

import DGS

image_file = '/mnt/c/Projects/AdvocateBeach2018/data/raw/images/test/FoxPointBeach/20180809_071703.jpg'

density = 10 # process every 10 lines
resolution = 0.01 # mm/pixel
dofilter =1 # filter the image
notes = 8 # notes per octave
maxscale = 8 #Max scale as inverse fraction of data length
verbose = 1 # print stuff to screen
x = -0.5
dgs_stats = DGS.dgs(image_file, density, resolution, dofilter, maxscale, notes, verbose, x)
