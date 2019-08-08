# How to Add Your Own Redistribution/Pruning/Growth Algorithms

This is a tutorial on how to add your own redistribution, pruning, and growth algorithms. The sparselearning library is built to be easily extendable in this regard. The basic steps are (1) implement your own function, pass it as an argument into the `Masking` class.

## General Structure of Functions

Here the general structure of the three functions:
```python
def your_redistribution(masking, name, weight, mask): return layer_importance
def your_growth(masking, name, new_mask, total_regrowth, weight): return new_mask
def your_pruning(masking, mask, weight, name): return pruned_mask
```

The variable `masking` is the general `Masking` class which enables access to global and local statistics of layers which can be useful to construct your own algorithms. `name` is the name of the current layer that is being processed. `weight` and `mask` are the weight of that layer and the binary mask that indicates the sparsity pattern. In the sparselearning library, all `0` elements in `mask` correspond to `0.0` values in `weight`.

## Accessible Variables

When you write the redistribution, growth, and pruning algorithms you will have access to the `Masking` class and the `name` of the current layer. This section gives you more details on what you can access and how.

### Access to the Optimizer

You can access the optimizer using the `masking.optimizer` variable. We can use this to for example get access to the momentum variables of the optimizer. This is for example how you can implement momentum redistribution used in the paper:
```python
def your_redistribution(masking, name, weight, mask):
    momentum = masking.optimizer.state[weight]['momentum_buffer']
    return momentum[mask.byte()].sum().item()

```

Other useful terms:
```python
# running adam sum of square (equivalent to RMSProp).
# Can be used to calculate the variance of gradient updates for each weight.
adam_sumsq = masking.optimizer.state[weight]['exp_avg_sq'] 
```

### Access to Global and Layer Statistics.

You can access statistics such as the number of non-zero weights of the current layer via the `masking` variable and the `name` of the layer. You have access to these statistics:
    Accessable global statistics:
```python

# Layer statistics:
non_zero_count = masking.name2nonzeros[name]
zero_count = masking.name2zeros[name]
normalized_layer_importance = masking.name2variance[name]
number_of_pruned_weights = masking.name2removed[name]
# Global Network statistics:
masking.total_nonzero
masking.total_zero
masking.total_removed
```

## Example: Variance-based Redistribution and Pruning

### Intuition

Here I added two example extensions for redistribution and pruning. These two examples look at the variance of the gradient. If we look at weights with high and low variance in their gradients over time, then we can have the following interpretations.

For high variance weights, we can have two perspectives. The first one would assume that weights with high variance are unable to model the interactions in the inputs to classify the outputs due to a lack of capacity. For example a weight might have a problem to be useful for both the digit 0 and digit 7 when classifying MNIST and thus has high variance between these examples. If we add capacity to high variance layers, then we should reduce the variance over time meaning the new weights can now fully model the different classes (one weight for 7 one weight for 0). According to this perspective we want to add more parameters to layers with high average variance. In other words, we want to redistribute pruned parameters to layers with high gradient variance.

The second perspective is a "potential of be useful" perspective. Here we see weights with high variance as having "potential to do the right classification, but they might just not have found the right decision boundary between classes yet". For example, a weight might have problems being useful for both the digit 7 and 0 but overtime it can find a feature which is useful for both classes. Thus gradient variance should reduce over time as features become more stable. If we take this perspective then it is important to keep some medium-to-high variance weights. Low variance weights have "settled in" and follow the gradient for a specific set of classes. These weights will not change much anymore while high variance weights might change a lot. So high variance weights might have "potential" while the potential of low variance weights is easily assessed by looking at the magnitude of that weights. Thus we might improve pruning if we look at both the variance of the gradient _and_ the magnitude of weights. You can find these examples in ['mnist_cifar/extensions.py']('sparse_learning/mnist_cifar/extensions.py').

### Implementation

```python
def variance_redistribution(masking, name, weight, mask):
    '''Return the mean variance of existing weights.

    Intuition: Higher gradient variance means a layer does not have enough
    capacity to model the inputs with the current number of weights.
    Thus we want to add more weights if we have higher variance.
    If variance of the gradient stabilizes this means
    that some weights might be useless/not needed.
    '''
    # Adam calculates the running average of the sum of square for us
    # This is similar to RMSProp. 
    if 'exp_avg_sq' not in masking.optimizer.state[weight]:
        print('Variance redistribution requires the adam optimizer to be run!')
        raise Exception('Variance redistribution requires the adam optimizer to be run!')
    iv_adam_sumsq = torch.sqrt(masking.optimizer.state[weight]['exp_avg_sq'])

    layer_importance = iv_adam_sumsq[mask.byte()].mean().item()
    return layer_importance


def magnitude_variance_pruning(masking, mask, weight, name):
    ''' Prunes weights which have high gradient variance and low magnitude.

    Intuition: Weights that are large are important but there is also a dimension
    of reliability. If a large weight makes a large correct prediction 8/10 times
    is it better than a medium weight which makes a correct prediction 10/10 times?
    To test this, we combine magnitude (importance) with reliability (variance of
    gradient).

    Good:
        Weights with large magnitude and low gradient variance are the most important.
        Weights with medium variance/magnitude are promising for improving network performance.
    Bad:
        Weights with large magnitude but high gradient variance hurt performance.
        Weights with small magnitude and low gradient variance are useless.
        Weights with small magnitude and high gradient variance cannot learn anything usefull.

    We here take the geometric mean of those both normalized distribution to find weights to prune.
    '''
    # Adam calculates the running average of the sum of square for us
    # This is similar to RMSProp. We take the inverse of this to rank
    # low variance gradients higher.
    if 'exp_avg_sq' not in masking.optimizer.state[weight]:
        print('Magnitude variance pruning requires the adam optimizer to be run!')
        raise Exception('Magnitude variance pruning requires the adam optimizer to be run!')
    iv_adam_sumsq = 1./torch.sqrt(masking.optimizer.state[weight]['exp_avg_sq'])

    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])

    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    max_var = iv_adam_sumsq[mask.byte()].max().item()
    max_magnitude = torch.abs(weight.data[mask.byte()]).max().item()
    product = ((iv_adam_sumsq/max_var)*torch.abs(weight.data)/max_magnitude)*mask
    product[mask==0] = 0.0

    x, idx = torch.sort(product.view(-1))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask
```

### Adding our extension to MNIST
To add our new methods to the MNIST script, we can simply import our newly created functions and define strings which enable our redistribution/pruning methods by passing a specific argument to the script:
```python
from extensions import magnitude_variance_pruning, variance_redistribution
if args.prune == 'magnitude_variance': args.prune = magnitude_variance_pruning
if args.redistribution == 'variance': args.redistribution = variance_redistribution
```

With this we can now run our new pruning and redistribution method by calling the script. However, our pruning method also requires the adam optimizer and thus we need to change the optimizer and the learning rate as well:
```bash
python main.py --model lenet5 --optimizer adam --prune magnitude_variance --redistribution variance --verbose --lr 0.001
```

Running 10 additional iterations (add `--iters 10`) of our new method with 5% weights on MNIST with Caffe LeNet-5 we can quickly calculate the performance using the evaluation script.
```bash
python get_results_from_logs.py 

Accuracy. Median: 0.99300, Mean: 0.99300, Standard Error: 0.00019, Sample size: 11, 95% CI: (0.99262,0.99338)
Error.    Median: 0.00700, Mean: 0.00700, Standard Error: 0.00019, Sample size: 11, 95% CI: (0.00662,0.00738)
Loss.     Median: 0.02200, Mean: 0.02175, Standard Error: 0.00027, Sample size: 11, 95% CI: (0.02122,0.02228)

```

Sparse momentum achieves an error of 0.0069 for this setting and the upper 95% confidence interval is 0.00739. Thus for this setting our results overlap with the confidence intervals of sparse momentum. Thus our new variance method is _as good_ as sparse momentum for this particular problem (Caffe LeNet-5 with 5% weights on MNIST).
