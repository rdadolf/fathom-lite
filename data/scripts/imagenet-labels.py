#!/usr/bin/env/python

import numpy as np

PATH='/fathom-lite/data/tmp'
# Take just the first 1000
labels = np.load(PATH+'/train_labels.npy')[0:1000]

print labels.shape

np.savez('imagenet-labels.npz', labels)
