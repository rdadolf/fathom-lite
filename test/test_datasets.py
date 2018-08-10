import numpy as np
import pytest

import fathomlite as fl

@pytest.fixture
def config_fix():
  config_file = fl.Config.find('test/test.config')
  if config_file is not None:
    fl.Config.load(config_file)
    print 'using config file:',config_file
  else:
    print 'no config file found'
  yield

@pytest.mark.parametrize(['dataset','shape'],
  [
    ['mnist-inputs',(1000,784)],
    ['mnist-labels',(1000,)],
    ['imagenet-inputs',(1000,224,224,3)],
    ['imagenet-labels',(1000,)],
    ['babi-stories',(1000,10,6)],
    ['babi-questions',(1000,6)],
    ['babi-answers',(1000,)],
  ])
class TestDatasets(object):
  def test_loadshapes(self, dataset, shape):
    dat = fl.DataLoader().load(dataset)
    assert dat.size>0
    assert list(dat.shape)==list(shape)

