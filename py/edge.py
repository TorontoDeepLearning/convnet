from util import *

def ChooseEdge(edge_proto):
  if edge_proto.edge_type == convnet_config_pb2.Edge.CONVOLUTIONAL:
    return ConvEdge(edge_proto)
  elif edge_proto.edge_type == convnet_config_pb2.Edge.CONV_ONETOONE:
    return ConvOneToOneEdge(edge_proto)
  elif edge_proto.edge_type == convnet_config_pb2.Edge.FC:
    return FCEdge(edge_proto)
  elif edge_proto.edge_type == convnet_config_pb2.Edge.MAXPOOL:
    return MaxPoolEdge(edge_proto)
  elif edge_proto.edge_type == convnet_config_pb2.Edge.RESPONSE_NORM:
    return ResponseNormEdge(edge_proto)
  else:
    raise Exception('Edge type not implemented.')

class Edge(object):
  def __init__(self, edge_proto):
    self.source_name_ = edge_proto.source
    self.dest_name_ = edge_proto.dest
    self.num_modules_ = 1
    self.name_ = '%s:%s' % (self.source_name_, self.dest_name_)

  def SetSource(self, l):
    self.source_ = l
    self.num_input_channels_ = l.GetNumChannels() 

  def SetDest(self, l):
    self.dest_ = l
    self.num_output_channels_ = l.GetNumChannels() 

  def GetSourceName(self):
    return self.source_name_

  def GetDestName(self):
    return self.dest_name_

  def GetSource(self):
    return self.source_

  def GetDest(self):
    return self.dest_

  def SetImageSize(self, image_size):
    self.image_size_ = image_size
    self.num_modules_ = 1

  def GetNumModules(self):
    return self.num_modules_

  def AllocateMemory(self):
    pass

  def LoadParams(self, f):
    pass

  def ComputeUp(self, input_layer, output_layer, overwrite):
    pass

class EdgeWithWeight(Edge):
  def __init__(self, edge_proto):
    super(EdgeWithWeight, self).__init__(edge_proto)
    self.weights_ = None
    self.bias_ = None

  def LoadParams(self, f):
    w_name = '%s:weight' % self.name_
    w = f[w_name].value.T
    assert self.weights_.shape == w.shape, "Shape mismatch %s %s %s" % (w_name, self.weights_.shape, w.shape)
    self.weights_.overwrite(w)
    b_name = '%s:bias' % self.name_
    b = f[b_name].value.reshape(1, -1)
    assert self.bias_.shape == b.shape, "Shape mismatch %s" % (b_name, self.bias_.shape, b.shape)
    self.bias_.overwrite(b)

class ConvEdge(EdgeWithWeight):
  def __init__(self, edge_proto):
    super(ConvEdge, self).__init__(edge_proto)
    self.kernel_size_ = edge_proto.kernel_size
    self.stride_ = edge_proto.stride
    self.padding_ = edge_proto.padding
    self.shared_bias_ = edge_proto.shared_bias

  def SetImageSize(self, image_size):
    self.image_size_ = image_size
    self.num_modules_ = (image_size + 2 * self.padding_
                         - self.kernel_size_) / self.stride_ + 1

  def AllocateMemory(self):
    input_size = self.kernel_size_**2 * self.num_input_channels_
    if self.shared_bias_:
      bias_locs = 1
    else:
      bias_locs = self.num_modules_**2
    if self.weights_ is not None:
      self.weights_.free_device_memory()
    if self.bias_ is not None:
      self.bias_.free_device_memory()
    self.weights_ = cm.empty((self.num_output_channels_, input_size))
    self.bias_ = cm.empty((1, self.num_output_channels_ * bias_locs))

  def ComputeUp(self, input_layer, output_layer, overwrite):
    scale_targets = 0 if overwrite else 1
    w = self.weights_
    b = self.bias_
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    batch_size = input_state.shape[0]
    cc.convUp(input_state, w, output_state, self.image_size_, self.num_modules_,
              self.num_modules_, self.padding_, self.stride_,
              self.num_input_channels_, scale_targets)
    if self.shared_bias_:
      output_state.reshape((-1, self.num_output_channels_))
    output_state.add_row_vec(b)
    if self.shared_bias_:
      output_state.reshape((batch_size, -1))

class MaxPoolEdge(Edge):
  def __init__(self, edge_proto):
    super(MaxPoolEdge, self).__init__(edge_proto)
    self.kernel_size_ = edge_proto.kernel_size
    self.stride_ = edge_proto.stride
    self.padding_ = edge_proto.padding

  def SetImageSize(self, image_size):
    self.image_size_ = image_size
    self.num_modules_ = (image_size + 2 * self.padding_
                         - self.kernel_size_) / self.stride_ + 1

  def ComputeUp(self, input_layer, output_layer, overwrite):
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    cc.MaxPool(input_state, output_state, self.num_input_channels_,
               self.kernel_size_, self.padding_, self.stride_,
               self.num_modules_)

class ResponseNormEdge(Edge):
  def __init__(self, edge_proto):
    super(ResponseNormEdge, self).__init__(edge_proto)
    self.num_filters_response_norm_ = 0
    self.blocked_ = edge_proto.response_norm_in_blocks
    self.add_scale_ = edge_proto.add_scale
    self.pow_scale_ = edge_proto.pow_scale
    self.frac_ = edge_proto.frac_of_filters_response_norm

  def SetImageSize(self, image_size):
    self.image_size_ = image_size
    self.num_modules_ = image_size
    self.num_filters_response_norm_ = int(self.frac_ * self.num_input_channels_)

  def ComputeUp(self, input_layer, output_layer, overwrite):
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    cc.ResponseNormCrossMap(input_state, output_state, self.num_input_channels_,
                            self.num_filters_response_norm_, self.add_scale_,
                            self.pow_scale_, self.blocked_)

class FCEdge(EdgeWithWeight):
  def __init__(self, edge_proto):
    super(FCEdge, self).__init__(edge_proto)

  def AllocateMemory(self):
    input_size = self.image_size_**2 * self.num_input_channels_
    self.weights_ = cm.empty((self.num_output_channels_, input_size))
    self.bias_ = cm.empty((1, self.num_output_channels_))

  def ComputeUp(self, input_layer, output_layer, overwrite):
    scale_targets = 0 if overwrite else 1

    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    w = self.weights_
    b = self.bias_
    cm.dot(input_state, w.T, target=output_state, scale_targets=scale_targets)
    output_state.add_row_vec(b)

class ConvOneToOneEdge(EdgeWithWeight):
  def __init__(self, edge_proto):
    super(ConvOneToOneEdge, self).__init__(edge_proto)

  def SetImageSize(self, image_size):
    self.image_size_ = image_size
    self.num_modules_ = image_size

  def AllocateMemory(self):
    self.weights_ = cm.empty((self.num_output_channels_,
                              self.num_input_channels_))
    self.bias_ = cm.empty((1, self.num_output_channels_))

  def ComputeUp(self, input_layer, output_layer, overwrite):
    scale_targets = 0 if overwrite else 1
    w = self.weights_
    b = self.bias_
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    batch_size = input_state.shape[0]
    input_state.reshape((-1, self.num_input_channels_))
    output_state.reshape((-1, self.num_output_channels_))
    cm.dot(input_state, w.T, target=output_state, scale_targets=scale_targets)
    output_state.add_row_vec(b)

    input_state.reshape((batch_size, -1))
    output_state.reshape((batch_size, -1))

