from data_handler import *
import datetime
import time
import matplotlib.pyplot as plt
plt.ion()
from time import sleep


def GetNormalizedImage(image, image_size_y, image_size_x):
  image = image.reshape(-1, image_size_y, image_size_x)
  num_colors = image.shape[0]
  image2 = np.zeros((image_size_y, image_size_x, num_colors), dtype=image.dtype)
  for i in xrange(num_colors):
    image2[:, :, i] = image[i, :, :]

  mx = image2.max()
  mn = image2.min()
  image2 = (255*(image2 - mn) / (mx-mn)).astype('uint8')
  return image2

def DisplayImages(image_list, image_size_y, image_size_x, fig=1):
  images = [GetNormalizedImage(image, image_size_y, image_size_x) for image in image_list]
  num_images = len(images)
  plt.figure(fig)
  plt.clf()
  for i in xrange(num_images):
    plt.subplot(1, num_images, i+1)
    plt.imshow(images[i], interpolation="nearest")
  plt.draw()

def GetNormalizedWeight(w, w_shape4d, r, c):
  """w: num_filters X sizeX**2 * num_colors."""
  num_output_channels, kernel_size_x, kernel_size_y, num_input_channels = w_shape4d
  image = np.zeros((kernel_size_y * r, kernel_size_x * c, num_input_channels))
  for i in range(r):
    for j in range(c):
      if i*c + j < num_output_channels:
        f = w[i*c + j, :].reshape(num_input_channels, kernel_size_y, kernel_size_x)
        for k in range(num_input_channels):
          image[i*kernel_size_y:(i+1)*kernel_size_y, j*kernel_size_x:(j+1)*kernel_size_x, k] = f[k, :, :]
  mx = image.max()
  mn = image.min()
  image = (255*(image - mn) / (mx-mn)).astype('uint8')
  return image

def DisplayWeights(weight_list, fig=2):
  num_images = len(weight_list)
  plt.figure(fig)
  plt.clf()
  for i in xrange(num_images):
    w, w_shape4d = weight_list[i]
    num_output_channels, kernel_size_x, kernel_size_y, num_input_channels = w_shape4d
    r = 8  #int(np.sqrt(numouts))
    c = (num_output_channels + r - 1) / r
    image = GetNormalizedWeight(w, w_shape4d, r, c)
    plt.subplot(1, num_images, i+1)
    plt.imshow(image, interpolation="nearest")
    color = 'k'
    ymax = kernel_size_y * r
    xmax = kernel_size_x * c
    for y in range(0, r):
      plt.axhline(y=y*kernel_size_y-0.5, xmin=0, xmax=xmax, color=color)
    for x in range(0, c):
      plt.axvline(x=x*kernel_size_x-0.5, ymin=0, ymax=ymax, color=color)
  plt.draw()


def Save(filename, data_dict):
  print 'Saving model to', filename
  f = h5py.File(filename, 'w')
  for key, value in data_dict.items():
    dset = f.create_dataset(key, value.shape, value.dtype)
    dset[:, :] = value
  f.close()

def Update(w, dw, dw_history, momentum, eps, l2_decay):
  dw_history.mult(momentum)
  if l2_decay != 0:
    dw.add_mult(w, l2_decay)
  dw_history.add_mult(dw, -eps)
  w.add(dw_history)

def Train(data_handle, model_filename):
  print_after = 10
  display_after = 5
  save_after = 100
  max_iter = 10000
  num_filters = 64
  kernel_size_y = 7
  kernel_size_x = 7
  stride_y = 2
  stride_x = 2
  padding_y = 1
  padding_x = 1
  momentum = 0.9
  epsilon = 0.0000001
  l2_decay = 0.001
  dropprob = 0.0
  noise_scale = 0.1
  batch_size, image_size_x, image_size_y, num_input_channels = data_handle.GetBatchShape()
  num_inputs = kernel_size_x * kernel_size_y * num_input_channels
  conv_desc = cm.GetConvDesc(num_input_channels, num_filters,
                             kernel_size_y, kernel_size_x, stride_y,
                             stride_x, padding_y, padding_x)
  images_shape = (batch_size, image_size_x, image_size_y, num_input_channels)
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)
  num_dims = image_size_y * image_size_x * num_input_channels

  v = cm.empty(images_shape)
  v_noise = cm.empty(images_shape)
  v_rec = cm.empty(images_shape)
  deriv_v = cm.empty(images_shape)
  w_enc = cm.CUDAMatrix(0.01 * (2 * np.random.rand(num_filters, num_inputs) - 1))
  w_dec = cm.CUDAMatrix(0.01 * (2 * np.random.rand(num_filters, num_inputs) - 1))
  b_enc = cm.CUDAMatrix(np.zeros((1, num_filters)))
  b_dec = cm.CUDAMatrix(np.zeros((1, num_inputs)))
  h = cm.empty(output_shape)
  deriv_h = cm.empty(output_shape)
  w_enc.set_shape4d(filters_shape)
  w_dec.set_shape4d(filters_shape)
  dw_enc = cm.empty(filters_shape)
  dw_dec = cm.empty(filters_shape)
  db_enc = cm.empty_like(b_enc)

  dw_enc_history = cm.empty(filters_shape)
  dw_dec_history = cm.empty(filters_shape)
  db_enc_history = cm.empty_like(b_enc)
  dw_enc_history.assign(0)
  dw_dec_history.assign(0)
  h.assign(0)
  deriv_h.assign(0)
  deriv_v.assign(0)
  v.assign(0)
  v_rec.assign(0)
  loss = 0
  for i in xrange(1, max_iter+1):
    sys.stdout.write('\r%d' % i)
    sys.stdout.flush()
   
    # Fprop.
    data_handle.GetBatch(v)
    v_noise.fill_with_randn()
    v_noise.mult(noise_scale)
    v_noise.add(v)
    cc_gemm.convUp(v_noise, w_enc, h, conv_desc)
    cc_gemm.AddAtAllLocs(h, b_enc)
    h.lower_bound(0)
    h.dropout(dropprob, scale=1.0/(1.0-dropprob))
    cc_gemm.convDown(h, w_dec, v_rec, conv_desc)

    # Compute Deriv.
    v_rec.subtract(v, target=deriv_v)
    loss += deriv_v.euclid_norm()**2 / (batch_size * num_dims)
    if i % print_after == 0:
      loss /= print_after
      sys.stdout.write(' Loss %.5f\n' % loss)
      loss = 0

    # Backprop.
    cc_gemm.convUp(deriv_v, w_dec, deriv_h, conv_desc)
    deriv_h.mult(1.0/(1.0- dropprob))
    deriv_h.apply_rectified_linear_deriv(h)
   
    cc_gemm.AddUpAllLocs(deriv_h, db_enc)
    cc_gemm.convOutp(v_noise, deriv_h, dw_enc, conv_desc)
    cc_gemm.convOutp(deriv_v, h, dw_dec, conv_desc)

    # Update weights.
    eps = float(epsilon)/batch_size
    Update(w_enc, dw_enc, dw_enc_history, momentum, eps, l2_decay)
    Update(w_dec, dw_dec, dw_dec_history, momentum, eps, l2_decay)
    Update(b_enc, db_enc, db_enc_history, momentum, eps, 0)
  
    if i % display_after == 0:
      DisplayImages([v.asarray()[0, :], v_noise.asarray()[0, :], v_rec.asarray()[0, :], deriv_v.asarray()[0, :]], image_size_y, image_size_x, fig=1)
      DisplayWeights([(w_enc.asarray(), w_enc.shape4d), (w_dec.asarray(), w_dec.shape4d)])

    if i % save_after == 0:
      Save(model_filename, {
        'w_enc' : w_enc.asarray(),
        'w_dec' : w_dec.asarray(),
        'b_enc' : b_enc.asarray(),
        'b_dec' : b_dec.asarray(),
      })

def main():
  st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
  data_filename = '/nobackup/nitish/imagenet/imagenet_train.h5'
  mean_filename = '/ais/gobi3/u/nitish/imagenet/pixel_mean.h5'
  model_filename = '/ais/gobi3/u/nitish/imagenet/models/conv_autonecoder_%s.h5' % st
  batch_size = 128
  data_handle = DataHandler(data_filename, mean_filename, 256, 256, 224, 224, batch_size, 128)
  Train(data_handle, model_filename)

if __name__ == '__main__':
  #pdb.set_trace()
  board = LockGPU()
  print 'Using board', board
  cm.CUDAMatrix.init_random(0)
  main()
  FreeGPU(board)
