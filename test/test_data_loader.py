import numpy as np
import pytest

import fathomlite as fl

class TestDataLoader(object):
  @pytest.mark.parametrize('length', [31,50,67,100,113])
  @pytest.mark.parametrize('batch_size', [1,2,3,7,13])
  @pytest.mark.parametrize('n_batches', [37,73,89])
  def test_batching(self, length, batch_size, n_batches):
    v = np.arange(0,length)
    batcher = fl.data_loader.Batcher(v)

    total = 0
    for i in xrange(0,n_batches):
      batch = batcher.next_batch(batch_size)
      assert len(batch)==batch_size, 'Incorrect batch size'
      total += len(batch)

    assert total==batch_size*n_batches, 'Incorrect total number of elements'
