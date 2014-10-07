Python interface to ConvNet

Dependencies
- h5py
- numpy
- protobuf

A simple example of running forward props through a ConvNet model -
```
python run_convnet.py <model_file(.pbtxt)> <model_parameters(.h5)> <means_file(.h5)>
```
For example,
```
python run_convnet.py ../examples/imagenet/CLS_net_20140801232522.pbtxt ../examples/imagenet/CLS_net_20140801232522.h5 ../examples/imagenet/pixel_mean.h5
```
