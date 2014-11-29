from google.protobuf import text_format
import convnet_config_pb2
import numpy as np
import os

def ReadLog(fnames):
  data = [np.loadtxt(fname, ndmin=2) for fname in fnames if os.path.exists(fname)]
  if len(data) == 0:
    return None
  else:
    return np.vstack(tuple(data))

def GetAllTimestamps(prefix, timestamp):
  timestamps = []
  fname = prefix + '_' + timestamp + '.pbtxt'
  try:
    model = convnet_config_pb2.Model()
    proto_pbtxt = open(fname, 'r')
    text_format.Merge(proto_pbtxt.read(), model)
    for timestamp in model.timestamp:
      timestamps.append(timestamp)
  except Exception as e:
    print 'Could not parse %s. May be an older verion.' % fname
    timestamps.append(timestamp)

  return timestamps

def ReadTrainLog(prefix, timestamp):
  timestamps = GetAllTimestamps(prefix, timestamp)
  fnames = [prefix + '_' + timestamp + '_train.log' for timestamp in timestamps]
  return ReadLog(fnames)

def ReadValLog(prefix, timestamp):
  timestamps = GetAllTimestamps(prefix, timestamp)
  fnames = [prefix + '_' + timestamp + '_valid.log' for timestamp in timestamps]
  return ReadLog(fnames)
