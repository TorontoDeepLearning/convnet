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

def CreateConvDesc(edge_proto):
  return cm.GetConvDesc(1, 1,
    edge_proto.kernel_size, edge_proto.kernel_size,
    edge_proto.stride, edge_proto.stride,
    edge_proto.padding, edge_proto.padding)

class Edge(object):
  def __init__(self, edge_proto):
    self.source_name_ = edge_proto.source
    self.dest_name_ = edge_proto.dest
    self.num_modules_y_ = 1
    self.num_modules_x_ = 1
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

  def SetImageSize(self, image_size_y, image_size_x):
    self.image_size_y_ = image_size_y
    self.image_size_x_ = image_size_x
    self.num_modules_y_ = 1
    self.num_modules_x_ = 1

  def GetNumModules(self):
    return self.num_modules_y_, self.num_modules_x_

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
    self.conv_desc_ = CreateConvDesc(edge_proto)
    self.shared_bias_ = edge_proto.shared_bias

  def SetImageSize(self, image_size_y, image_size_x):
    self.conv_desc_.num_input_channels = self.num_input_channels_
    self.conv_desc_.num_output_channels = self.num_output_channels_
    self.image_size_y_ = image_size_y
    self.image_size_x_ = image_size_x
    self.num_modules_y_, self.num_modules_x_ = cm.GetOutputShape(image_size_y, image_size_x, self.conv_desc_)

  def AllocateMemory(self):
    input_size = self.conv_desc_.kernel_size_x * self.conv_desc_.kernel_size_y * self.num_input_channels_
    if self.shared_bias_:
      bias_locs = 1
    else:
      bias_locs = self.num_modules_y_ * self.num_modules_x_
    if self.weights_ is not None:
      self.weights_.free_device_memory()
    if self.bias_ is not None:
      self.bias_.free_device_memory()
    self.weights_ = cm.empty((self.num_output_channels_, input_size))
    self.bias_ = cm.empty((1, self.num_output_channels_ * bias_locs))
    self.weights_.set_shape4d(
      (self.num_output_channels_, self.conv_desc_.kernel_size_x,
       self.conv_desc_.kernel_size_y, self.num_input_channels_))

  def ComputeUp(self, input_layer, output_layer, overwrite):
    scale_targets = 0 if overwrite else 1
    w = self.weights_
    b = self.bias_
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    batch_size = input_state.shape[0]
    cc_gemm.convUp(input_state, w, output_state, self.conv_desc_, scale_targets)
    if self.shared_bias_:
      output_state.reshape((-1, self.num_output_channels_))
    output_state.add_row_vec(b)
    if self.shared_bias_:
      output_state.reshape((batch_size, -1))

class MaxPoolEdge(Edge):
  def __init__(self, edge_proto):
    super(MaxPoolEdge, self).__init__(edge_proto)
    self.conv_desc_ = CreateConvDesc(edge_proto)

  def SetImageSize(self, image_size_y, image_size_x):
    self.conv_desc_.num_input_channels = self.num_input_channels_
    self.conv_desc_.num_output_channels = self.num_output_channels_
    self.image_size_y_ = image_size_y
    self.image_size_x_ = image_size_x
    self.num_modules_y_, self.num_modules_x_ = cm.GetOutputShape(image_size_y, image_size_x, self.conv_desc_)

  def ComputeUp(self, input_layer, output_layer, overwrite):
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    cc_gemm.MaxPool(input_state, output_state, self.conv_desc_)

class ResponseNormEdge(Edge):
  def __init__(self, edge_proto):
    super(ResponseNormEdge, self).__init__(edge_proto)
    self.num_filters_response_norm_ = 0
    self.blocked_ = edge_proto.response_norm_in_blocks
    self.add_scale_ = edge_proto.add_scale
    self.pow_scale_ = edge_proto.pow_scale
    self.frac_ = edge_proto.frac_of_filters_response_norm

  def SetImageSize(self, image_size_y, image_size_x):
    self.image_size_y_ = image_size_y
    self.image_size_x_ = image_size_x
    self.num_modules_y_ = image_size_y
    self.num_modules_x_ = image_size_x
    self.num_filters_response_norm_ = int(self.frac_ * self.num_input_channels_)

  def ComputeUp(self, input_layer, output_layer, overwrite):
    input_state = input_layer.GetState()
    output_state = output_layer.GetState()
    cc_gemm.ResponseNormCrossMap(
      input_state, output_state, self.num_filters_response_norm_, 
      self.add_scale_, self.pow_scale_, self.blocked_)

class FCEdge(EdgeWithWeight):
  def __init__(self, edge_proto):
    super(FCEdge, self).__init__(edge_proto)

  def AllocateMemory(self):
    input_size = self.image_size_x_ * self.image_size_y_ * self.num_input_channels_
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

  def SetImageSize(self, image_size_y, image_size_x):
    self.image_size_y_ = image_size_y
    self.image_size_x_ = image_size_x
    self.num_modules_y_ = image_size_y
    self.num_modules_x_ = image_size_x

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
