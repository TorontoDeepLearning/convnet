- Download the data from www.cs.toronto.edu/~nitish/data/mnist.h5 [390Mb]
- Set the data_dir in all *_data.pbtxt so that it points to the directory where
  the data was downloaded. 
- Set the checkpoint directory in net.pbtxt. This is where the model will write
  out checkpoints. Make sure this diretory has been created.

Run:
train_convnet <board-id> net.pbtxt train_data.pbtxt val_data.pbtxt

Toronto users-
Make sure the board is locked.
