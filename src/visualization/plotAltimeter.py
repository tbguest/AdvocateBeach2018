#!/usr/bin/python3


"""
  Richard A. Cheel
"""

import numpy as np
import pylab as plt

def main():
  # main code here
  fn = 'rng10152018_pi71.001'
  dn = 'C:\Projects\AdvocateBeach2018\data\raw\range_data\15_10_2018\PM\'

  data = np.genfromtxt(dn+fn,delimiter='R',usecols=(1,))
  plt.figure
  plt.plot(data)
  plt.xlabel('index')
  plt.ylabel('Range [mm]')
  plt.show()   


if __name__ == "__main__":
  main()

