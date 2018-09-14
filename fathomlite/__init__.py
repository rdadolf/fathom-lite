from configuration import Config

filepath=None
try:
  filepath = Config.find()
except IOError: # Allow failure
  pass
else:
  Config.load(filepath)
del filepath

from data_loader import DataLoader,Batcher

#from deepq import *
#from speech import *
#from seq2seq import *
from .autoenc import Autoenc, AutoencFwd
from .memnet import MemNet, MemNetFwd
from .alexnet import AlexNet, AlexNetFwd
from .vgg import VGG, VGGFwd
from .residual import Residual, ResidualFwd
