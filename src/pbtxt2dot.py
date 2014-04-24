""" Script for rendering a network into a dot file. """
# python pbtxt2dot.py CLS_net_multigpu image.dot
# dot -Tpng image.dot -o outfile.png

import sys
import os
import re
from google.protobuf import text_format
import convnet_config_pb2

def ReadModel(proto_file):
  protoname, ext = os.path.splitext(proto_file)
  proto = convnet_config_pb2.Model()
  proto_pbtxt = open(proto_file, 'r')
  txt = proto_pbtxt.read().replace(';', '')
  text_format.Merge(txt, proto)
  return proto

def GetSizes(model):
  size_dict = {}
  patch_size = model.patch_size
  for l in model.layer:
    if l.is_input:
      size = patch_size
    else:
      e = next(e for e in model.edge if e.dest == l.name)
      if e.edge_type == convnet_config_pb2.Edge.FC:
        size = 1
      elif e.edge_type == convnet_config_pb2.Edge.RESPONSE_NORM:
        input_size = size_dict[e.source]
        size = input_size
      else:
        input_size = size_dict[e.source]
        size = (input_size + 2 * e.padding - e.kernel_size)/ e.stride + 1
    size_dict[l.name] = size
  return size_dict

def main():
  model = ReadModel(sys.argv[1])
  output = open(sys.argv[2], 'w')
  size_dict = GetSizes(model)
  output.write('digraph G {\n')

  for l in model.layer:
    size = size_dict[l.name]
    txt = "%s\\n %d - %d - %d" % (l.name, size, size, l.num_channels)
    output.write('%s [shape=box, label = "%s"];\n' % (l.name, txt))
  for e in model.edge:
    output.write('%s -> %s [dir="back"];\n' % (e.dest, e.source))
  output.write('}\n')
  output.close()

if __name__ == '__main__':
  main()
