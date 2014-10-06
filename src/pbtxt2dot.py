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

  for l in proto.layer:
    num_channels = l.num_channels
    for ls in l.layer_slice:
      num_channels += ls.num_channels
    l.num_channels = num_channels
  return proto

def AddSubnet(model, subnet):
  submodel = ReadModel(subnet.model_file)
  for s in submodel.subnet:
    AddSubnet(submodel, s)
  name = subnet.name
  merge_layers = {}
  remove_layers = []
  for merged_layer in subnet.merge_layer:
    merge_layers[merged_layer.subnet_layer] = merged_layer.net_layer
  for l in subnet.remove_layer:
    remove_layers.append(l)
  for layer in submodel.layer:
    if layer.name not in merge_layers and layer.name not in remove_layers:
      l = model.layer.add()
      l.CopyFrom(layer)
      l.name = name + "_" + layer.name
      l.num_channels = layer.num_channels * subnet.num_channels_multiplier  
      l.gpu_id = layer.gpu_id + subnet.gpu_id_offset
  for edge in submodel.edge:
    if edge.source in remove_layers or edge.dest in remove_layers:
      continue
    e = model.edge.add()
    e.CopyFrom(edge)
    e.gpu_id = e.gpu_id + subnet.gpu_id_offset
    if edge.source in merge_layers:
      e.source = merge_layers[edge.source]
    else:
      e.source = name + "_" + edge.source
    if edge.dest in merge_layers:
      e.dest = merge_layers[edge.dest]
    else:
      e.dest = name + "_" + edge.dest

def SetIO(model):
  dest_layers = []
  source_layers = []
  for edge in model.edge:
    dest_layers.append(edge.dest)
    source_layers.append(edge.source)
  for layer in model.layer:
    layer.is_input = layer.name not in dest_layers
    layer.is_output = layer.name not in source_layers
  for layer in model.layer:
    print layer.name, layer.is_input, layer.is_output


def Sort(model):
  S = []
  L = []
  outgoing_edge = {}
  incoming_edge = {}
  mark = {}
  for l in model.layer:
    outgoing_edge[l.name] = [e for e in model.edge if e.source == l.name]
    incoming_edge[l.name] = [e for e in model.edge if e.dest == l.name]
  for e in model.edge:
    mark[GetName(e)] = False

  for l in model.layer:
    if l.is_input:
      S.append(l)
  while len(S) > 0:
    n = S.pop()
    L.append(n)
    for e in outgoing_edge[n.name]:
      mark[GetName(e)] = True
      m = next(l for l in model.layer if l.name == e.dest)
      if reduce(lambda x, y: x and y, [mark[GetName(ee)] for ee in incoming_edge[m.name]], True):
        S.append(m)
  return L


def GetSizes(model):
  size_dict = {}
  patch_size = model.patch_size
  L = Sort(model)
  for l in L:
    if l.is_input:
      size = patch_size
    else:
      e = next(e for e in model.edge if e.dest == l.name)
      source_name = e.source
      if e.tied_to != "":
        e = next(ee for ee in model.edge if e.tied_to == '%s:%s' % (ee.source, ee.dest))
      if e.edge_type == convnet_config_pb2.Edge.FC:
        size = 1
      elif e.edge_type == convnet_config_pb2.Edge.RESPONSE_NORM:
        input_size = size_dict[source_name]
        size = input_size
      elif e.edge_type == convnet_config_pb2.Edge.CONV_ONETOONE:
        input_size = size_dict[source_name]
        size = input_size
      elif e.edge_type == convnet_config_pb2.Edge.DOWNSAMPLE:
        input_size = size_dict[source_name]
        size = input_size / e.sample_factor
      elif e.edge_type == convnet_config_pb2.Edge.UPSAMPLE:
        input_size = size_dict[source_name]
        size = input_size * e.sample_factor
      else:
        input_size = size_dict[source_name]
        size = (input_size + 2 * e.padding - e.kernel_size)/ e.stride + 1
    size_dict[l.name] = size
  return size_dict

def main():
  model = ReadModel(sys.argv[1])
  for subnet in model.subnet:
    AddSubnet(model, subnet)
  SetIO(model)
  size_dict = GetSizes(model)
  
  output = open(sys.argv[2], 'w')
  output.write('digraph G {\n')

  show_gpu = True
  for l in model.layer:
    size = size_dict[l.name]
    txt = "%s\\n %d - %d - %d" % (l.name, size, size, l.num_channels)
    if show_gpu:
      txt += " (%d)" % l.gpu_id
    output.write('%s [shape=box, label = "%s"];\n' % (l.name, txt))
  for e in model.edge:
    if e.tied_to:
      color = "blue"
    else:
      color = "black"
    txt = "(%d)" % (e.gpu_id)
    output.write('%s -> %s [dir="back", color=%s, label="%s"];\n' % (e.dest, e.source, color, txt))
  output.write('}\n')
  output.close()

if __name__ == '__main__':
  main()
