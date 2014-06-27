### Running the pre-trained model
Download the model [476Mb]
```
wget http://www.cs.toronto.edu/~nitish/models/CLS_net_20140621074703.h5
```

Run the test images

- Batch-mode on a GPU.
```
$ extract_representation <board-id> <model-file> <data-config-file> <output> <layer-names>
```
For example,
```
$ extract_representation 0 CLS_net_20140621074703.pbtxt test_images.pbtxt output.h5 hidden7 output
$ python show_results.py output.h5
```
This should produce an output like sample_output.txt

- One-at-a-time on a CPU only.
Run make in the convnet/cpu directory.
```
$ extract_representation_cpu <model-file> <model-parameters> <pixel-mean> <output-dir> <layer-names>  < <image-files>
```
For example,
```
$ extract_representation_cpu CLS_net_20140621074703.pbtxt CLS_net_20140621074703.h5 pixel_mean.h5 cpu_out hidden7 output  < test_images.txt
$ python show_results.py cpu_out/output.txt
```
This will take each image in test_images.txt at write the features for layers
hidden7 and output into text files in the cpu_out directory.
