import matplotlib.pyplot as plt
import os
import glob

homechar = "C:\\"

tide = "tide19"
position = "position2"
#vidspec = "vid_1540304255" # pos1
vidspec = "vid_1540307860" # pos2

imgdir = os.path.join(homechar, "Projects", "AdvocateBeach2018", "data", "interim", \
                      "images", "fromVideo", tide, position, vidspec)

imgs = sorted(glob.glob(os.path.join(imgdir, 'img*.jpg')))
img = plt.imread(imgs[200]) # use default color
