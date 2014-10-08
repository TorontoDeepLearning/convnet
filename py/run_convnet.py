import sys
import convnet as cn
import numpy as np
import Image

def LoadImage(file_name, resize=256, crop=224):
  image = Image.open(file_name)
  width, height = image.size

  if width > height:
    width = (width * resize) / height
    height = resize
  else:
    height = (height * resize) / width
    width = resize
  left = (width  - crop) / 2
  top  = (height - crop) / 2
  image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
  # image_resized.show()
  # raw_input('Press Enter.')
  data = np.array(image_resized.getdata()).T
  if len(data.shape) == 1:
    data = np.tile(data, (3, 1))
  return data.reshape(1, -1)

def Usage():
  print 'python run_convnet.py <model_file(.pbtxt)> <model_parameters(.h5)> <means_file(.h5)>'

def main():
  board = cn.LockGPU()
  print 'Using board', board
  if len(sys.argv) < 4:
    Usage()
    sys.exit(1)
  pbtxt_file = sys.argv[1]
  params_file = sys.argv[2]
  means_file = sys.argv[3]
  model = cn.ConvNet(pbtxt_file)
  model.Load(params_file)
  model.SetNormalizer(means_file, 224)
  print model.GetLayerNames()
  
  # Random inputs.
  data = np.random.randn(128, 224 * 224 * 3)
  model.Fprop(data)
  output = model.GetState('output')
  print output

  # Load image.
  image_data = LoadImage('../examples/imagenet/test_images/0.jpg')
  model.Fprop(image_data)
  last_hidden_layer = model.GetState('hidden7')
  output = model.GetState('output')
  print output.argmax()

  cn.FreeGPU(board)

if __name__ == '__main__':
  main()
