#!/usr/bin/env python

import numpy as np
import sys

if __name__=='__main__':
  if len(sys.argv)<2:
    print('Usage: '+sys.argv[0]+' <datafile.npz> ...')

  for filename in sys.argv[1:]:
    d = np.load(filename)
    print(filename+': '+str(d['arr_0'].shape))
