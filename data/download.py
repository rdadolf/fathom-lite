#!/usr/bin/env python

import argparse
import os.path
import shutil
import sys
import urllib2

FL_URL='http://rdadolf.com/hosted/fathom-lite/data'
VERSION='1.0'

DATA_FILES=[
  'mnist-inputs.npz',
  'mnist-labels.npz',
  'imagenet-inputs.npz',
  'imagenet-labels.npz',
]
  

class NetworkError(Exception): pass
class MissingDataError(Exception): pass


def check_host_has_data(url):
  file_url = url+'/version'
  response = urllib2.urlopen(file_url)
  version_string = response.read().strip()

  if version_string!=VERSION:
    raise MissingDataError('Server alive but does not appear to have up-to-date Fathom-lite data.')

  return True

def download_if_not_cached(file_url, file_dest, force=False):
  if os.path.isfile(file_dest) and not force:
    print 'File cache found: '+file_dest
    return False # File cached. (To force download, just remove the file.)
  dir = os.path.dirname(os.path.abspath(file_dest))
  if not os.path.isdir(dir):
    raise IOError('Destination directory "'+str(dir)+'" does not exist')

  try:
    print 'Downloading '+file_url
    response = urllib2.urlopen(file_url)
    with open(file_dest,'wb') as f:
      shutil.copyfileobj(response,f)
  except urllib2.HTTPError:
    print 'Error when downloading '+file_url
    raise

  return file_dest


def get_options():
  cli = argparse.ArgumentParser('Download Fathom-lite data files from the Internet.')

  cli.add_argument('-f','--force',
                   default=False, action='store_true',
                   help='Ignore file caches')
  cli.add_argument('-n','--no-version',
                   default=False, action='store_true',
                   help='Do not check server for data version number')
  cli.add_argument('-d','--dir',
                   default='.', type=str,
                   help='Download data to alternate directory.')
  return cli.parse_args()


if __name__=='__main__':
  opts = get_options()

  if not opts.force:
    status = check_host_has_data(FL_URL)

  for filename in DATA_FILES:
    file_url = FL_URL+'/'+filename
    file_dest = os.path.normpath(opts.dir+'/'+filename)
    download_if_not_cached(file_url, file_dest, force=opts.force)


