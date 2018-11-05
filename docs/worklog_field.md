# Advocate 2018
## Daily notes and field work log

### Bob Taylor's benchmark:

#### Pre-localization
  - 358151.5327,5022552.7000,4.8790,TBAR

#### post-localization
- day 1 (day of setup): 358163.1746,5022550.0901,4.8883,TBAR
- day 2: 358163.1700,5022550.0579,4.9089,,  
- oct18: 358163.1836,5022550.0536,4.9349
- oct19: 358163.1833,5022550.0714,4.9087
- oct20: 358163.1808,5022550.0455,4.9255
- oct21-1: 358163.1814,5022550.0544,4.9186
- oct21-2: 358163.1841,5022550.0609,4.9328
- oct22-1: 358163.1982,5022550.0601,4.8886
- oct22-2: 358163.1658,5022550.0593,4.9335
- oct23-1: 358163.1721,5022550.0762,4.9116
- oct24-1: 358163.1614,5022550.0490,4.8996 (before)
- oct24-1: 358163.1835,5022550.0764,4.9106 (after)
- oct25-1: 358163.1692,5022550.0576,4.9037
- oct25-2: 358163.1980,5022550.0719,4.9058
- oct26-1: 358163.1842,5022550.0517,4.9019
- oct26-2: 358163.1772,5022550.0505,4.9075
- oct27-1: 358163.1957,5022550.0390,4.8910

#### NB. thoughts on size sorting at ADV beach wrt energy:
On coarsening during low energy, fining during high energy, and orgainization of material at top of beach...
- low energy: surf and swash zones are more closely tied -- part of the same system; bore collapse mobilizes all sediment, swash moves it around
- high energy: surf and swash zones are "decoupled"; the wide (20+ m) swash zone provides time for coarse to stop saltating, but fines are carried far from bore collapse. I.e. longer settling lag.

Why is coarse material favoured during fair weather? WHere do the fines go?


### Sunday, 14 Oct
- arrived ~16:30
- Strong NW wind
- setup, no data collection

### Mon, 15 Oct
- clear morning, light wind
- cusps; 5-8 m
- setup RTK
- sieved, painted stones.
  - largest sieve size (look into), 2nd, 3rd, 4th largest
  - green, yellow, orange, not painted - respectively
- established survey grid-1
- opted to wait until Wed for pT deployment -- strong wind forecast for Tues
- midday low tide: surveys
  - RTK: cross-shore lines -25, 0; longshore lines -5 and 5
  - camera: cross-shore line 0; longshore line -5
- Afternoon high tide: data collection
  - no SfM
  - both frames put out just in time
  - no timestamp logged with range data
  - green, yellow, orange stones deployed
  - array frame moved seaward for ~last 5 mins - no rtk marks on new positions
- cusps
- remember to add post height to camera case RTK points!

### Tues, 16 Oct
- Strong overnight wind (35 kts+)
- AM survey
  - RTK + camera: cross-shore line 0; longshore lines -15, -5, 5
- new grid setup
- Overhead video discovered to be dropping frames. suspect high cpu load while writing to usb (strange though)

### Wed, 17 Oct
- light wind at first light. Deployed overhead frame after high tide to test new code for dropped frames (still dropping frames)
- to check frame count for video, use '''ffprobe -show_streams -count_frames -pretty .\vid_1539399711.h264''' and look for nb_read_frames
- midday survey (new JOB - sparse grid + dense grid)
  - RTK + camera: cross-shore line 0; longshore line -5 (3m spaced); dense grid 1
- still troubleshooting vid problems. trying network solution  


### Thurs, 18 Oct
- lowered survey camera - tripod mount
- midday survey (new JOB - sparse grid + dense grid)
  - RTK + camera: cross-shore line 0; longshore line -5 (3m spaced); dense grid 1
- deployed pT frame
- still troubleshooting vid problems. overclocking seems to work - frame rate is nearly constant, but not quite (e.g. O(10s) of frames / 14400)
- see ADVOCATE-GRID-OCT18.txt for pT frame coords

### Fri, 19 Oct
- windy (W, 20-30 km/h)
- cobble cam in the am - test purposes
- used vivitar cam as well
- midday survey  

### Sat, 20 Oct
- 9 am ADT: setup array frame and logged during early ebb. one position only.
- remember to add post height to RTK pi positions - saved in data-interim-GPS-20_10_2018
- windy (20-30 km/h from S-SW)
- lots of froth in swash tongue (check for wonky sonar returns)
- might note coarsening associated with top of swash (coarse migration landward). 4 sensors othwise redundant
- survey - ~11:30 am -- dense grid 1 (last day of this grid)
- Marie-Carmen and Francisco arrived yesterday, left today

### Sun, 21 Oct
- AM - light wind from N-NW
- first light survey (dense grid 2 - first day)
- set up frames immediately after
- SfM photos at assumed logging site (too high)
- cusps began to form with late flood. Frames logging just to landward
- Moved array frame twice before settling on a position. Pi locations logged
- see logbook for a couple notes on timing
- 3 cobble cam frame positions, each with GCPs
- looking like a good dataset

### Mon, 22 Oct
- sunrise survey
- wind NW shifting to W, picking up, ~25 to 30 km/h
- small waves at sunup, building as wind shifted to W
- huge buildup of round, coarse pebbles at high tide, with horn and bermlike features. O(30cm) height.
- surveyed densearray2 and longshore 1

- set up frame in bay -- between berm/horns -- so width of swash zone would be limited by the steeper slopes on either side. Could not connect to pis. Suspect eth cable issue. Pulled out - no data.

- afternoon survey - 1500 ADT
- beach planar and coarse, patchy
- wind 30 km/h W

### Tues, 23 Oct
- light w wind at sunrise
- survey
- set up both frames. wind backing off by noon
- see notes in lab book

### Wed, 24 Oct
- light wind offshore
- set up array frame only after survey on sharp, narrow berm
- tide inundated berm
- most data will be sea surface time series
- small, incipient cusps

### Thurs, 25 Oct
- AM survey
- wet snow, wind NW, becoming W 30 km/h
- sea building
- small wavelength cusps - 2-4 m
- took samples from 2 horns, 2 bays
- wind strengthening through day from west, developed sea -- no sampling
- richard left
- late afternoon survey -- cold hands!

### Fri, 26 Oct
- Blowing strong from the west-NW
- AM and PM surveys, otherwise working on defence presentation

### Sat, 27 Oct
- am survey -- light offshore wind
- set up Frames
- gals arrived
- couldn't talk to pi61, set up monoprice on tripod instead
- berm growth under array 

### Sun, 28 Oct
