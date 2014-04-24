- Download the data from www.cs.toronto.edu/~nitish/data/mnist.h5 [390Mb]
- Set the data_dir in all *_data.pbtxt so that it points to the directory where
  the data was downloaded. 
- Set the checkpoint directory in net.pbtxt. This is where the model, error
  stats, logs etc will be written. Make sure this diretory has been created.

Run:
$ train_convnet <board-id> net.pbtxt train_data.pbtxt val_data.pbtxt
or
$ train_convnet <board-id> net.pbtxt train_plus_val_data.pbtxt test_data.pbtxt

Toronto users-
Make sure the board is locked before running this.

This will train the net for 100,000 updates and write the model out in the
checkpoint directory. This takes 4-5 min (on NVIDIA Titan, may vary depending on
GPU). Training on train_plus_val_data.pbtxt should get around 1.2% test error by
then.

--------------------------
Looking at the performance
--------------------------
The *.log files in the checkpoint directory contain the performance metrics for
the training and validation sets.
Run
$ python show_plots.py

--------------------------
Draw the net.
--------------------------
$ python ../../src/pbtxt2dot.py net.pbtxt net.dot
$ dot -Tpng net.dot -o net.png

