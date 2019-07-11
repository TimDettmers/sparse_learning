# Sparse Learning Library and Sparse Momentum Resources

This repo contains a sparse learning library which allows you to wrap any PyTorch neural network with a sparse mask to emulate the training of sparse neural networks. It also contains the code to replicate our work [Sparse Networks from Scratch: Faster Training without Losing Performance](https://arxiv.org/abs/1907.04840).

## Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/).
2. Install other dependencies: `pip install -r requirements.txt`
3. Install the sparse learning library: `python setup.py install`

## Basic Usage

### MNIST & CIFAR-10 models

MNIST and CIFAR-10 code can be found in the `mnist_cifar` subfolder. You can run `python main.py --data DATASET_NAME --model MODEL_NAME` to run a model on MNIST (`--data mnist`) or CIFAR-10 (`--data cifar`).

The following models can be specified with the `--model` command out-of-the-box:
```
 MNIST:

	lenet5
	lenet300-100

 CIFAR-10:

	alexnet-s
	alexnet-b
	vgg-c
	vgg-d
	vgg-like
	wrn-28-2
	wrn-22-8
	wrn-16-8
	wrn-16-10
```

Beyond standard parameters like batch-size and learning rate which usage can be seen by `python main.py --help` the following sparse learning specific parameter are available:
```
--save-features       Resumes a saved model and saves its feature data to
                      disk for plotting.
--bench               Enables the benchmarking of layers and estimates
                      sparse speedups
--growth GROWTH       Growth mode. Choose from: momentum, random, and
                      momentum_neuron.
--death DEATH         Death mode / pruning mode. Choose from: magnitude,
                      SET, threshold.
--redistribution REDISTRIBUTION
                      Redistribution mode. Choose from: momentum, magnitude,
                      nonzeros, or none.
--death-rate DEATH_RATE
                      The pruning rate / death rate.
--density DENSITY     The density of the overall sparse network.
--sparse              Enable sparse mode. Default: True.

```

### Running an ImageNet Model

The ImageNet code for sparse momentum can be found in the sub-folder `imagenet` which contains an adjusted version of [NVIDIA Deep Learning Examples: RN50v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/RN50v1.5). For now, please follow the instructions of this repo (I will update more specific instructions later). You can use the adjusted `RN50_FP32_4GPU.sh` script in the `imagenet/example` folder to run an ImageNet example once you setup everything as described in the original repo.

### Running Your Own Model

With the sparse learning library it is easy to run sparse momentum on your own model. All that you need to do is follow the following code template:

![alt text][template]


## Extending the Library

Some changes to the library are still pending which will enable the easy extension with your own sparse learning library. Stay tuned.

[template]: https://timdettmers.com/wp-content/uploads/2019/07/code.png "Generic example usage of sparse learning library."
