""" Handles data."""
from util import *
def GetSizeString(m):
  return '%.3f MB' % ((m.shape[0] * m.shape[1] * 4) / (1024 * 1024.0))

class DataHandler(object):
  def __init__(self, data_filename, mean_filename, raw_image_size_y, raw_image_size_x, image_size_y, image_size_x, batch_size, chunk_size):
    self.data_file_ = h5py.File(data_filename, 'r')
    mean_file = h5py.File(mean_filename, 'r')
    self.dset_ = self.data_file_['data']
    mean = mean_file['pixel_mean'].value.reshape(-1, 1)
    std  = mean_file['pixel_std'].value.reshape(-1, 1)

    numpixels = image_size_y * image_size_x

    self.mean_ = cm.CUDAMatrix(np.tile(mean, (1, numpixels)).reshape(1, -1))
    self.std_  = cm.CUDAMatrix(np.tile(std , (1, numpixels)).reshape(1, -1))
    self.num_colors_ = mean.shape[0]


    self.dataset_size_, raw_numdims = self.dset_.shape
    assert raw_image_size_x * raw_image_size_y * self.num_colors_ == raw_numdims
    assert image_size_y <= raw_image_size_y
    assert image_size_x <= raw_image_size_x
    

    print 'Image size %d %d' % (image_size_y, image_size_x)
    self.data_gpu_buffer_ = cm.empty((raw_numdims, chunk_size))
    print "data buffer size: ", GetSizeString(self.data_gpu_buffer_)
    self.x_offset_ = cm.empty((1, batch_size))
    self.y_offset_ = cm.empty((1, batch_size))
    self.flip_ = cm.empty((1, batch_size))

    self.start_ = 0
    self.chunk_start_ = 0
    self.batch_size_ = batch_size
    self.chunk_size_ = chunk_size
    self.raw_image_size_y_ = raw_image_size_y
    self.raw_image_size_x_ = raw_image_size_x
    self.image_size_y_     = image_size_y
    self.image_size_x_     = image_size_x

  def GetBatchSize(self):
    return self.batch_size_

  def GetNumDims(self):
    return self.image_size_y_ * self.image_size_x_ * self.num_colors_
  
  def GetBatchShape(self):
    return self.batch_size_, self.image_size_x_, self.image_size_y_ , self.num_colors_

  def GetBatch(self, v):
    end = self.start_ + self.batch_size_
    if end > self.chunk_size_:
      self.start_ = 0
      end = self.batch_size_

    if self.start_ == 0:
      #if self.chunk_start_ == 0:
      #  Shuffle()
      data_cpu = self.dset_[self.chunk_start_:self.chunk_start_ + self.chunk_size_, :]
      self.data_gpu_buffer_.overwrite(data_cpu.T)
      self.chunk_start_ += self.chunk_size_
      if self.chunk_start_ > self.dataset_size_:
        self.chunk_start_ = 0

    self.x_offset_.fill_with_rand()
    self.y_offset_.fill_with_rand()
    self.flip_.fill_with_rand()
    self.x_offset_.mult_by_scalar(self.raw_image_size_x_- self.image_size_x_)
    self.y_offset_.mult_by_scalar(self.raw_image_size_y_ - self.image_size_y_)
    cm.extract_patches(self.data_gpu_buffer_.slice(self.start_, end), v,
                       self.x_offset_, self.y_offset_, self.flip_,
                       self.raw_image_size_x_, self.raw_image_size_y_,
                       self.image_size_x_, self.image_size_y_)
    #data_gpu_buffer.slice(start, end).transpose(target=v)
    v.add_row_mult(self.mean_, -1)
    v.div_by_row(self.std_)
    return v
