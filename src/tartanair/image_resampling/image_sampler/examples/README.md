# Examples for Image Sampler

## Generic Sampler Example: Optical Flow.
A generic sampler is one that samples data from one camera model and converts it to another. The new camera model might also be rotated, it's amazing! Seriously though, that's not even it. There's more. The input images could be preprocessed (with a provided function) prior to being sampled. And also postprocessed, after being sampled. 

As you may imagine, sampling optical flow images (that are represented as a 2-channel tensor) is not trivial. To help you out, we provide an example. To run the example, start with creating a directory and cloning both `image_sampler` and `mvs_utils` into it. Let's call it `image_resampling`. Then, run the following commands:

```bash
$ cd /the/top/of/your/directory/that/you/just/made/and/called/image_resampling
$ git clone https://github.com/castacks/image_sampler.git
$ git clone https://github.com/castacks/mvs_utils.git
$ cd mvs_utils
$ git submodule update --init --recursive
$ cd ../..
$ python3 -m image_resampling.image_sampler.examples.generic_sampler_optical_flow_sampling_example -v

```


