# Some notes on getting python kernel + atom up and running with opencv

Today, I tooled up on atom packages to improve my python programming workflow and switch from Spyder.

Using conda, I was able to get opencv working, but only using a python2 kernel.

Following the instructions found on the forums, I broke my usual rule of not using pip, and used pip (within a clean conda environment) to install the latest, python 3.7-compatable version of opencv.

I used:

~~~
conda create -n computervision python=3
conda activate computervision
~~~

**Note: I actually did away with this separate env approach, because I needed the packages that already existed in Advocate2018 env. The same works for the existing env, I just had to use py3.5 specific wheel files.**

I downloaded the most recent wheel files for numpy and opencv from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv], then (after navigating to dowloads folder)

~~~
pip install filename-opencv
pip install filename-numpy
~~~

Then, in order for my atom-hydrogen setup to be able to run the code inline, I needed to install in ipython kernel:

~~~
conda install ipykernel
python -m ipykernel install
~~~

Badabing. All set.
