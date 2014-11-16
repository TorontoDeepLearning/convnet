### Training
- Download the data from www.cs.toronto.edu/~nitish/data/mnist.h5 [390Mb]
- Set the data_dir in all *_data.pbtxt so that it points to the directory where
  the data was downloaded. 
- Set the checkpoint directory in net.pbtxt. This is where the model, error
  stats, logs etc will be written. Make sure this diretory has been created.

Run:
```
$ train_convnet --board=<board-id> --model=net.pbtxt --train=train_data.pbtxt --val=val_data.pbtxt
```
or
```
$ train_convnet --board=<board-id> --model=net.pbtxt --train=train_plus_val_data.pbtxt --val=test_data.pbtxt
```

Toronto users-
Make sure the board is locked before running this.

This will train the net for 100,000 updates and write the model out in the
checkpoint directory. This takes 4-5 min (on NVIDIA Titan, may vary depending on
GPU). Training on train_plus_val_data.pbtxt should get around 1.2% test error by
then.

### Extracting features
The representation at different layers of the learned model can be extracted:
```
$ extract_representation --board=<board-id> --model=<model-file> --feature-config=<feature-config-file>
```

For example, 
```
$ extract_representation --board=0 --model=checkpoint_dir/mnist_net_20140627130044.pbtxt --feature-config=feature_config.pbtxt
```
The `feature_config.pbtxt` file describes what data to input, which layers to extract features from and where to write the output.

### Looking at the performance
The *.log files in the checkpoint directory contain the performance metrics for
the training and validation sets.
Run
```
$ python ../../apps/show_plots.py ./checkpoint_dir
```

### Draw the net.
Draw the network graph.
```
$ python ../../apps/pbtxt2dot.py net.pbtxt net.dot
$ dot -Tpng net.dot -o net.png
```
