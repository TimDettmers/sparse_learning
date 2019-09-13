
## Release v0.1: Ease of use, bug fixes, and documentation.
### Bug fixes:
 - Fixed a but where magnitude pruning pruned too many parameters when the weight was dense (>95% density) and the pruning rate was small (<5%).
   First experiments on LeNet-5 Caffe indicate that this change did not affect performance for networks that learn to have dense weights.
   I will replicate this across architectures to make sure this bugfix does not change performance.
 - Fixed instabilities in SET (sparse evolutionary training) pruning which could cause nan values in specific circumstances.

### Documentation:
 - Added basic docstring documentation

### Features:
  - MNIST/CIFAR: Separate log files are not created for different models/densities/names.
  - MNIST/CIFAR: Aggregate mean test accuracy with standard errors can now be automatically extracted from logs with `python get_results_from_logs.py`.

### API:
  - Changed names from "death" to "prune" to be more consistent with the terminology in the paper.
  - Added --verbose argument to print the parameter distribution before/after pruning at the end of each epoch. By default, the pruning distribution will no longer be printed.
  - Removed --sparse flag and added --dense flag. The default is args.dense==False and thus sparse mode is enabled by default. To run a dense model just pass the --dense argument.


## Release v0.2: FP16 support, modularity of prune/growth/redistribution algorithms.

### Bug fixes:
 - Fixed a bug where global pruning would throw an error if a layer was fully dense and had a low prune rate.

### Features:
 - Added FP16 support. Any model can now be run in 16-bit by passing the [apex](https://github.com/NVIDIA/apex) `FP16_Optimizer` into the `Masking` class and replacing `loss.backward()` with `optimizer.backward(loss)`.
 - Added adapted [Dynamic Sparse Reparameterization](https://arxiv.org/abs/1902.05967) [codebase](https://github.com/IntelAI/dynamic-reparameterization) that works with sparse momentum.
 - Added modular architecture for growth/prune/redistribution algorithms which is decoupled from the main library. This enables you to write your own prune/growth/redistribution algorithms without touched the library internals. A tutorial on how to add your own functions was also added: [How to Add Your Own Algorithms](How_to_add_your_own_algorithms.md).



## Release v1.0: Bug fixes, New ImageNet Baselines

### Bug fixes:
 - Changed to boolean indexing for PyTorch 1.2 compatibility.
 - Fixed an error where an error can occur for global pruning algorithms if very few weights were removed for a layer.
 - Removed momentum reset. This feature did not have any effect on performance and made the algorithm more complex.
 - Fixed an error where two layers of VGG16 were removed by use of the `remove_weight_partial_name()` function. Results were slightly degraded, but weights needed for dense performance and relative ordering compared to other methods remained the same.

### Features:
 - Evaluation script can now aggregate log files organized in a folder hierarchy. For each folder results will be aggregated.
 - Added decay schedule argument. One can choose between Linear and Cosine prune rate decay schedules.
 - Added new ImageNet baseline which is based on the codebase of [Mostafa & Wang, 2019](https://arxiv.org/abs/1902.05967).
 - Added a max-thread argument which can be used to set the total maximum data loader threads for training, validation and test set data loaders.
