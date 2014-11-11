from edge import *

def ChooseLayer(layer_proto):
  if layer_proto.activation == convnet_config_pb2.Layer.LINEAR:
    return Layer(layer_proto)
  elif layer_proto.activation == convnet_config_pb2.Layer.RECTIFIED_LINEAR:
    return ReLULayer(layer_proto)
  elif layer_proto.activation == convnet_config_pb2.Layer.SOFTMAX:
    return SoftmaxLayer(layer_proto)
  else:
    raise Exception('Layer type not implemented.')

class Layer(object):
  def __init__(self, layer_proto):
    self.num_channels_ = layer_proto.num_channels
    self.is_input_ = True
    self.is_output_ = True
    self.incoming_edge_ = []
    self.outgoing_edge_ = []
    self.image_size_y_ = layer_proto.image_size_y
    self.image_size_x_ = layer_proto.image_size_x
    self.name_ = layer_proto.name
    self.dropprob_ = layer_proto.dropprob
    self.dropout_scale_up_at_train_time_ = True
    self.gaussian_dropout_ = layer_proto.gaussian_dropout
    self.state_ = None

  def GetName(self):
    return self.name_

  def GetNumChannels(self):
    return self.num_channels_

  def IsInput(self):
    return self.is_input_

  def SetSize(self, image_size_y, image_size_x):
    self.image_size_y_ = image_size_y
    self.image_size_x_ = image_size_x

  def GetSize(self):
    return self.image_size_y_, self.image_size_x_

  def AllocateMemory(self, batch_size):
    layer_size = self.num_channels_ * self.image_size_y_ * self.image_size_x_
    if self.state_ is not None:
      self.state_.free_device_memory()
    self.state_ = cm.empty((batch_size, layer_size))
    self.state_.set_shape4d((batch_size, self.image_size_x_, self.image_size_y_, self.num_channels_))
    self.state_.assign(0)

  def GetState(self):
    return self.state_

  def AddIncomingEdge(self, e):
    self.incoming_edge_.append(e)
    self.is_input_ = False
  
  def AddOutgoingEdge(self, e):
    self.outgoing_edge_.append(e)
    self.is_output_ = False

  def ApplyActivation(self):
    pass

  def ApplyDropout(self):
    if self.dropprob_ > 0 and not self.dropout_scale_up_at_train_time_ \
       and not gaussian_dropout_:
      # Scale down.
      self.state_.mult(1 - self.dropprob_)

class ReLULayer(Layer):
  def __init__(self, layer_proto):
    super(ReLULayer, self).__init__(layer_proto)

  def ApplyActivation(self):
    self.state_.lower_bound(0)
    self.ApplyDropout()

class SoftmaxLayer(Layer):
  def __init__(self, layer_proto):
    super(SoftmaxLayer, self).__init__(layer_proto)

  def ApplyActivation(self):
    self.state_.apply_softmax_row_major()
    self.ApplyDropout()
