""" Python implementation of forward props for ConvNet models."""
from layer import *

class ConvNet(object):
  def __init__(self, model_pbtxt):
    self.model_ = convnet_config_pb2.Model()
    proto_pbtxt = open(model_pbtxt, 'r')
    text_format.Merge(proto_pbtxt.read(), self.model_)
    self.layer_name_dict_ = {}
    self.BuildNet()
    self.normalizer_set_ = False
    self.batch_size_ = 0

  def BuildNet(self):
    self.layer_ = []
    self.edge_ = []
    L = self.Sort()  # Topological sort.
    for l in L:
      self.layer_.append(ChooseLayer(l))
    for e in self.model_.edge:
      self.edge_.append(ChooseEdge(e))
   
    for l in self.layer_:
      for e in self.edge_:
        l_name = l.GetName()
        
        if l_name == e.GetSourceName():
          l.AddOutgoingEdge(e)
          e.SetSource(l)

        if l_name == e.GetDestName():
          l.AddIncomingEdge(e)
          e.SetDest(l)

    for l in self.layer_:
      if l.IsInput():
        image_size = l.GetSize()
      else:
        # Incoming edge num_modules should be set because self.layer_ is sorted.
        image_size = l.incoming_edge_[0].GetNumModules()
      l.SetSize(image_size)

      for e in l.outgoing_edge_:
        e.SetImageSize(image_size)

    for l in self.layer_:
      self.layer_name_dict_[l.GetName()] = l

  def Sort(self):
    def GetName(edge):
      return '%s:%s' % (edge.source, edge.dest)

    model = self.model_
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

  def Load(self, params_file):
    f = h5py.File(params_file)
    for e in self.edge_:
      e.AllocateMemory()
      e.LoadParams(f)
    f.close()

  def SetBatchSize(self, batch_size):
    self.batch_size_ = batch_size
    for l in self.layer_:
      l.AllocateMemory(self.batch_size_)

  def Fprop(self, input_data):
    batch_size = input_data.shape[0]
    if self.batch_size_ != batch_size:
      self.SetBatchSize(batch_size)

    for l in self.layer_:
      overwrite = True
      for e in l.incoming_edge_:
        e.ComputeUp(e.GetSource(), l, overwrite)
        overwrite = False
      if l.IsInput():
        state = l.GetState()
        state.overwrite(input_data)
        self.Normalize(state)
        l.ApplyDropout()
      else:
        l.ApplyActivation()
        
  def GetLayerNames(self):
    return [l.GetName() for l in self.layer_]

  def GetState(self, layer_name):
    return self.layer_name_dict_[layer_name].GetState().asarray()

  def SetNormalizer(self, means_file, image_size=1):
    f = h5py.File(means_file)
    mean = f['pixel_mean'].value.reshape(1, -1)
    std  = f['pixel_std'].value.reshape(1, -1)
    self.mean_ = cm.CUDAMatrix(np.tile(mean, (image_size**2, 1)))
    self.std_  = cm.CUDAMatrix(np.tile(std,  (image_size**2, 1)))
    self.mean_.reshape((1, -1))
    self.std_.reshape((1, -1))
    self.normalizer_set_ = True

  def Normalize(self, state):
    if self.normalizer_set_:
      state.add_row_mult(self.mean_, -1)
      state.div_by_row(self.std_)
