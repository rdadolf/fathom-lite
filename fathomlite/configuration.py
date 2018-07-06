# Allows per-user configuration settings, specified in JSON files.
#
# I got tired of the /home/<username>/... and sys.path hacks, so I wrote
# up a replacement that allows those kind of variables to be specified external
# to the source code.

import copy
import os
import json
import types

class Config(object):
  '''Allows per-user, module-wide configuration settings, specified in JSON files.

  For module users:
    Most use is transparent, but you should know the priority used in searching
    for configuration files:
      - explicit filename (in the module code)
      - path given by the "CONFIGFILE" environment variable.
      - a "<project>.conf" file in the current directory
      - a ".<project>.conf" file in your home directory

  For module writers:
    Config gives a simple, module-wide way of accessing configuration settings.
    The most common usage is to find and load once at launch:
      ``Config.load(Config.find(<optional-filename>))``
    and then to lookup config keys by name:
      ``_ = Config.get('keyname')``
  '''

  # Class attributes
  ##############################################################################
  # REPLACE THIS NAME WITH YOUR PROJECT NAME.
  CONFIG_NAME='fathom-lite'
  ##############################################################################
  # These are the actual stored values
  _conf = dict()
  _path = None

  # Configuration files are stored in JSON. In order to enforce encodability,
  # only the following types are allowed.
  # https://docs.python.org/2/library/json.html#py-to-json-table
  _CONVERTIBLE_VALID_TYPES = [
    types.DictType,
    types.TupleType,
    types.ListType,
    types.StringType,
    types.UnicodeType,
    types.IntType,
    types.LongType,
    types.FloatType,
    types.BooleanType,
    types.NoneType,
  ]

  def __init__(self):
    assert False, 'Do not instantiate this class. Use class methods directly.'

  @classmethod
  def _validate(cls, obj):
    assert type(obj) in cls._CONVERTIBLE_VALID_TYPES, 'Value cannot be converted to JSON.'

  @classmethod
  def set(cls, key, value=True):
    cls._validate(key)
    cls._validate(value)
    cls._conf[key] = value
    return cls._conf[key]

  @classmethod
  def get(cls, key, default=None):
    cls._validate(key)
    try:
      v = cls._conf[key]
    except KeyError:
      if default is None:
        raise KeyError, '\'%s\' not found in configuration file \'%s\'.'%(key,cls._path)
      else:
        return default
    return v

  @classmethod
  def purge(cls):
    cls._conf = dict()
    cls._path = None

  @classmethod
  def load(cls, filename):
    cls._conf = json.load(open(filename))
    cls._path = filename

  @classmethod
  def save(cls, filename):
    json.dump(cls._conf, open(filename,'w'), indent=2)

  @classmethod
  def find(cls, filename=None):
    '''Looks in common locations for a configuration file.

    The location priority is:
      - A provided filename
      - Location given by a "CONFIGFILE" environment variable.
      - "<project>.conf" in current directory.
      - ".<project>.conf" in home directory.
    '''

    files_attempted = []

    v = filename
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v
    v = os.environ.get('CONFIGFILE',None)
    files_attempted.append('$CONFIGFILE')
    if v is not None and os.path.isfile(v):
      return v
    v = cls.CONFIG_NAME+'.conf'
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v
    v = os.path.expanduser('.'+cls.CONFIG_NAME+'.conf')
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v

    if filename is None:
      raise IOError('No valid configuration file found. Locations tried:\n'+'\n'.join(['  '+str(v) for v in files_attempted]))
