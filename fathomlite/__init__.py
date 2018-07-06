from configuration import Config

filepath=None
try:
  filepath = Config.find()
except IOError: # Allow failure
  pass
else:
  Config.load(filepath)
del filepath

#from deepq import *
#from speech import *
#from seq2seq import *
from .autoenc import Autoenc, AutoencFwd
#from memnet import *
#from alexnet import *
#from vgg import *
#from residual import *
