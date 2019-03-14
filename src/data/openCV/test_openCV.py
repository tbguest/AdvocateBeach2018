
# be sure to switch to conda env: "computervision" which has opencv setup with python2.7

import os
import glob
import numpy as np
import cv2 as cv
import matplotlib
# %matplotlib qt5
%matplotlib inline
import matplotlib.pyplot as plt

homechar = "C:\\"

tide = "tide19"
position = "position2"
#vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2

imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                      "images", "fromVideo", tide, position, vidspec)

imgs = sorted(glob.glob(os.path.join(imgdir, 'img*.jpg')))
img = cv.imread(imgs[200], 1) # use default color

b, g, r = cv.split(img)
rgb_split = np.concatenate((b,g,r), axis=1)

cv.namedWindow("Image", cv.WINDOW_NORMAL)
cv.imshow("Image", img)
cv.moveWindow("Image",0,0)

height,width,channels = img.shape
# cv.imwrite("output.jpg", img)


cv.namedWindow("Channels", cv.WINDOW_NORMAL)
cv.imshow("Channels", rgb_split)
cv.moveWindow("Channels",0,0)

#imS = cv.resize(rgb_split, (960, 1620))                    # Resize image

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)
hsv_split = np.concatenate((h,s,v), axis=1)

cv.namedWindow("Split HSV", cv.WINDOW_NORMAL)
cv.imshow("Split HSV", hsv_split)
#cv.moveWindow("Channels",0,1000)

cv.waitKey(0)
cv.destroyAllWindows()

plt.figure(1)
plt.imshow(img)
