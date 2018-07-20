#!/usr/bin/env/python

# As you may have noticed, this does not process raw ImageNet files.
# The image bounding box and preprocessing takes a while, and we've
# already done it several times on a cluster. This just re-uses the
# cached data and reformats it.

import hickle
import numpy as np

PATH='/fathom-lite/data/tmp'
_hkl = [hickle.load(PATH+'/000'+str(i)+'.hkl') for i in range(0,10)]
# Take just the first 1000 and swap to array of images rather than of channels
hkl = np.moveaxis(np.concatenate(_hkl)[:1000], [0,1,2,3],[0,3,1,2])

print hkl.shape

np.savez('imagenet-inputs.npz', hkl)
