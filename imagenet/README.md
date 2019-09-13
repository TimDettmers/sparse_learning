## Models for ImageNet

For ImageNet there are two different ResNet-50 models. The "baseline" model replicated results by [Mostafa & Wang, 2019](https://arxiv.org/abs/1902.05967) and is an adapted version of [IntelAI/dynamic-reparameterization](https://github.com/IntelAI/dynamic-reparameterization). This ImageNet model attains a 74.9% accuracy for a dense baseline. This model can be found in the `baseline` folder and is the main model used throughout the paper for comparison against other models on ImageNet.

The second version is adapted from [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/RN50v1.5). It is the same model but better tuned than the baseline above. This model uses warmup learning rates and label smoothing and has a baseline 77.0% which is in line with recent tuned ResNet-50 baselines (see [Saining et al., 2019](https://arxiv.org/abs/1904.01569). This codebase can be found in the `tuned` folder.

