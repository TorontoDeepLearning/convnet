Python interface to ConvNet

Dependencies
- h5py
- numpy
- protobuf

A simple example of running forward props through a ConvNet model -
```
  import convnet as cn
  ...
  model = cn.ConvNet(pbtxt_file)  # Load the model architecture.
  model.Load(params_file)  # Set the weights and biases.
  model.SetNormalizer(means_file, 224)  # Set the mean and std for input normalization.
  
  data = np.random.randn(128, 224 * 224 * 3)  # 128 images of size 224x224 as a numpy array.
  model.Fprop(data)  # Fprop through the model.
  
  # Returns the state of the requested layer as a numpy array.
  last_hidden_layer = model.GetState('hidden7')
  output = model.GetState('output')
  
  print output.shape, last_hidden_layer.shape  # (128, 1000) (128, 4096).
```


```
python run_convnet.py <model_file(.pbtxt)> <model_parameters(.h5)> <means_file(.h5)>
```
For example,
```
python run_convnet.py ../examples/imagenet/CLS_net_20140801232522.pbtxt ../examples/imagenet/CLS_net_20140801232522.h5 ../examples/imagenet/pixel_mean.h5
```
