# Sparse Learning Library and Sparse Momentum Resources

This repo contains a sparse learning library which allows you to wrap any PyTorch neural network with a sparse mask to emulate the training of sparse neural networks. It also contains the code to replicate our work [Sparse Networks from Scratch: Faster Training without Losing Performance](https://arxiv.org/abs/1907.04840).

## Requirements

The library requires PyTorch v1.2. You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. For CUDA version < 9.2 you need to either compile from source, or install a new CUDA version along with a compatible video driver.

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

To run ImageNet with 16-bit you need to install [Apex](https://github.com/NVIDIA/apex). For me it currently does not work to install apex from pip, but installing it from the repo works just fine.

The ImageNet code for sparse momentum can be found in the sub-folder `imagenet` which contains two different ResNet-50 ImageNet models: A baseline that is used by Mostafa & Wang (2019) which reaches 74.9% accuravy with 100% weights and a tuned ResNet-50 version which is identical to the baseline but uses a warmup learning rate and label smoothing and reaches 77.0% accuracy with 100% weights. The tuned version builds on [NVIDIA Deep Learning Examples: RN50v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/RN50v1.5) while the baseline builds on [Intel/dynamic-reparameterization](https://github.com/IntelAI/dynamic-reparameterization). 

### Running Your Own Model

With the sparse learning library it is easy to run sparse momentum on your own model. All that you need to do is follow the following code template:

![alt text][template]


## Extending the Library

It is easy to extend the library with your own functions for growth, redistribution and pruning. See [The Extension Tutorial](https://github.com/TimDettmers/sparse_learning/blob/master/How_to_add_your_own_algorithms.md) for more information about how you can add your own functions.

[template]: https://timdettmers.com/wp-content/uploads/2019/07/code.png "Generic example usage of sparse learning library."
