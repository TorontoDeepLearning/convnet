### Running the pre-trained model
Download the model [476Mb]
```
wget http://www.cs.toronto.edu/~nitish/models/CLS_net_20140621074703.h5
```

There are two ways to extract features -

- **One-at-a-time on a CPU.**

Run make in the convnet/cpu directory. This should produce a binary called `extract_representation_cpu` in convnet/bin. Then the run the `extract_representation_cpu` binary as follows -
```
$ extract_representation_cpu <model-file> <model-parameters> <pixel-mean> <output-dir> <layer-names>  < <image-files>
```
For example,
```
$ extract_representation_cpu CLS_net_20140621074703.pbtxt CLS_net_20140621074703.h5 pixel_mean.h5 cpu_out hidden7 output  < test_images.txt
```
This will take each image in `test_images.txt` at write out the features at layers
`hidden7` and `output` into text files in the `cpu_out` directory. `hidden7` are the top-level features and `output` is the distribution over the 1000 ILSVRC 2013 categories.
The names of other layers can be found in `CLS_net_20140621074703.pbtxt`.

To see the classification results-
```
$ python show_results.py cpu_out/output.txt
```


- **Batch-mode on a GPU.**

After running make in the convnet/ directory -
```
$ extract_representation <board-id> <model-file> <data-config-file> <output> <layer-names>
```
For example,
```
$ extract_representation 0 CLS_net_20140621074703.pbtxt test_images.pbtxt output.h5 hidden7 output
$ python show_results.py output.h5
```
This should produce an output like [sample_output.txt](https://github.com/TorontoDeepLearning/convnet/blob/master/examples/imagenet/sample_output.txt) The test images are from the [Toronto Deep Learning Classification Demo](http://deeplearning.cs.toronto.edu/)

