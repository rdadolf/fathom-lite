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

@pytest.mark.parametrize('modelname',
  ['Autoenc', 'AlexNet'])
 # ['Seq2Seq','MemNet','Speech','Autoenc','Residual','VGG','AlexNet','DeepQ'])
class TestInstantiation(object):

  def _models(self, modelname):
    train_model = getattr(fl,modelname)
    inference_model = getattr(fl,modelname+'Fwd')
    return train_model, inference_model

  def test_instantiate(self, config_fix, modelname):
    Model,ModelFwd = self._models(modelname)
    m = Model()
    m = ModelFwd()

  def test_setup(self, config_fix, modelname):
    Model,ModelFwd = self._models(modelname)
    for m in [Model(), ModelFwd()]:
      m.setup()

  def test_run(self, config_fix, modelname):
    Model,ModelFwd = self._models(modelname)
    for m in [Model(), ModelFwd()]:
      m.setup()
      m.run()
