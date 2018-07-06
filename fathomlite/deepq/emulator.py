# NOTE: Tejas Kulkarni's implementation
import os.path
import numpy as np
import sys
from ale_python_interface import ALEInterface
import cv2
import time

ROM_PATH = 'fathom/deepq/roms/'

class emulator(object):
  def __init__(self, rom_name, vis,frameskip=1,windowname='preview'):
    self.ale = ALEInterface()
    self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
    self.ale.setInt("random_seed",123)
    self.ale.setInt("frame_skip",frameskip)
    romfile = str(ROM_PATH)+str(rom_name)
    if not os.path.exists(romfile):
      print 'No ROM file found at "'+romfile+'".\nAdjust ROM_PATH or double-check the filt exists.'
    self.ale.loadROM(romfile)
    self.legal_actions = self.ale.getMinimalActionSet()
    self.action_map = dict()
    self.windowname = windowname
    for i in range(len(self.legal_actions)):
      self.action_map[self.legal_actions[i]] = i

    # print(self.legal_actions)
    self.screen_width,self.screen_height = self.ale.getScreenDims()
    print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
    self.vis = vis
    if vis:
      cv2.startWindowThread()
      cv2.namedWindow(self.windowname, flags=cv2.WINDOW_AUTOSIZE) # permit manual resizing

  def get_image(self):
    numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
    self.ale.getScreenRGB(numpy_surface)
    image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
    return image

  def newGame(self):
    self.ale.reset_game()
    return self.get_image()

  def next(self, action_indx):
    reward = self.ale.act(action_indx)
    nextstate = self.get_image()
    # scipy.misc.imsave('test.png',nextstate)
    if self.vis:
      cv2.imshow(self.windowname,nextstate)
      if sys.platform == 'darwin':
        # if we don't do this, can hang on OS X
        cv2.waitKey(2)
    return nextstate, reward, self.ale.game_over()



if __name__ == "__main__":
  engine = emulator('breakout.bin',True)
  engine.next(0)
  time.sleep(5)
