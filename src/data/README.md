Comment headers need to be added to all these scripts.

## Grain sizing

### 1. Survey images

### 2. PiImages
- Manually curate image sets; delete images containing swash, seaweed, shadows, etc. in the image region to be processed.
- crop images in matlab (crop_images.m)
- run image set through DGS pipeline for images with variable height (process_grain_images_variable_range.py). This must be done via the Ubuntu shell on my machine, since problems with package dependencies have not allowed me to successfully install the DGS package alongside anaconda.
- run consolidate_digital_grain_size_and_range_output.py for relevant tide to organize data into more accessible format. i.e. chunks associated with each position (this assumes some work has already been done with the range data, I think).

## Cobble tracking

From raw video:
- decimate video into frame with ffmpeg:
`ffmpeg -i vid_1540311466.h264 img%06d.jpg -hide_banner`
- move images to relevant folder and separate nonrelevant images (manually)
- run time averaging code 'cobble_tracking_time_average.py' (set the nimgs parameter based on desired amount of averaging)
- turn time averaged images back into a video to use with the tracker:
`ffmpeg -r 25 -f image2 -start_number 710 -i img%06d.jpg position4_10frameavg.webm`
- use 'opencv_cobble_tracker.py' to track stones (one at a time). Initialize tracker over stone, and reinitialize if/when the tracker fails. Use the full, unaveraged image set to help match up trajectories. The code needs to be altered after each run to update the cobble number (and change video, location data, etc.). Code is run from the command line using:
`python opencv_cobble_tracker.py --video C:\Projects\AdvocateBeach2018\data\interim\images\timeAverage\tide19\position3\vid_1540307860\tenframeaverage\position3_10frameavg.webm --tracker kcf`
- extract pixel-to-distance scaling using extract_scaling_from_cobble_images.py. This requires a relevant targets photo to the manually selected and put in the 'targets' folder
- use scaling and smoothed, scaled pressure record to define mean and +/- extents of swash zone, and some prelim visualization in sin_fit_swash_timeseries.py
