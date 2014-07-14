### Running the pre-trained model
Download the model [476Mb]
```
wget http://www.cs.toronto.edu/~nitish/models/CLS_net_20140621074703.h5
```

There are two ways to extract features -

- **One-at-a-time on a CPU.**

Run make in the convnet/cpu directory. This should produce a binary called `extract_representation_cpu` in convnet/bin. Then the run the `extract_representation_cpu` binary as follows -
```
$ extract_representation_cpu --model <model-file> --parameters <model-parameters> --mean <pixel-mean> --output <output-dir> --layer <layer-name> [--layer <layer-name> ..]  < <image-files>
```
For example,
```
$ extract_representation_cpu --model CLS_net_20140621074703.pbtxt --parameters CLS_net_20140621074703.h5 --mean pixel_mean.h5 --output cpu_out --layer hidden7 --layer output  < test_images.txt
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
$ extract_representation --board <board-id> --model <model-file> --feature-config <feature-config-file>
```
For example,
```
$ extract_representation --board 0 --model CLS_net_20140621074703.pbtxt --feature-config feature_config.pbtxt
$ python show_results.py output.h5
```
This should produce an output like [sample_output.txt](https://github.com/TorontoDeepLearning/convnet/blob/master/examples/imagenet/sample_output.txt) The test images are from the [Toronto Deep Learning Classification Demo](http://deeplearning.cs.toronto.edu/)

To average over different patches (center + 4 corners) * 2 (horizontal flip)
```
$ extract_representation --board 0 --model CLS_net_20140621074703.pbtxt --feature-config feature_config_avg10.pbtxt
$ python show_results.py output_avg10.h5
```


