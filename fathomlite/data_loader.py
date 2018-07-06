import numpy as np
import os
import os.path

from configuration import Config


# Simple class to encapsulate data loading from files.
# This mostly exists to centralize the code in one place, in case I
# need to change things later.
class DataLoader(object):
  # Helpful constants for Fathom-lite data.
  number_of_samples = 1000

  def __init__(self, data_dir='data'):
    '''DataLoader is designed to pull sample data from the fathom-data repo:
      github.com/rdadolf/fathom-data

      The data_dir option is the location of the cloned fathom-data repo.'''
    self.data_dir = Config.get('data_dir',data_dir)

  def load(self, filename):
    '''Loads data from a .npz file.

    The filename should be the name of the .npz file, relative to the fathom-data
    repository specified when the DataLoader object was created. If the filename
    does not end with ".npz", that suffix will be added.'''

    if not filename.endswith('.npz'):
      # All fathom-data files end with .npz
      filename += '.npz'
    path = os.path.normpath(os.path.join(self.data_dir, filename))
    if not os.path.isfile(path):
      print 'Current directory is',os.getcwd()
      raise IOError('No such data file: '+str(path))
    npz_data = np.load(path)
    if 'arr_0' not in npz_data:
      raise KeyError('No "arr_0" element in data file "'+str(os.path.normpath(path))+'". Are you sure this file is Fathom-lite data?')
    return npz_data['arr_0']
