#!/usr/bin/env python

import numpy as np

if __name__=='__main__':
  images = np.fromfile('t10k-images-idx3-ubyte',dtype='uint8')[16:]
  labels = np.fromfile('t10k-labels-idx1-ubyte',dtype='uint8')[8:]

  images = images[:784000].reshape((1000,784))
  labels = labels[:1000]

  np.savez('mnist-inputs.npz',images)
  np.savez('mnist-labels.npz',labels)

