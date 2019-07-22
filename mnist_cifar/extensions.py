import math
import torch

# Through the masking variable we have access to the following variables/statistics.
'''
    Access to optimizer:
        masking.optimizer

    Access to momentum/Adam update:
        masking.get_momentum_for_weight(weight)

    Accessable global statistics:

    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]

    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
'''

def your_redistribution(masking, name, weight, mask):
    '''
    Returns:
        Layer importance      The unnormalized layer importance statistic
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.
    '''
    return layer_importance

#===========================================================#
#                         EXAMPLE                           #
#===========================================================#
def variance_redistribution(masking, name, weight, mask):
    '''Return the mean variance of existing weights.

    Higher variance means the layer does not have enough
    capacity to model the inputs with the number of current weights.
    If weights stabilize this means that some weights might
    be useless/not needed.
    '''
    layer_importance = torch.var(weight.grad[mask.byte()]).mean().item()
    return layer_importance


def your_pruning(masking, mask, weight, name):
    """Returns:
        mask        Pruned Binary mask where 1s indicated active
                    weights. Can be modified in-place or newly
                    constructed
    """
    return mask

#===========================================================#
#                         EXAMPLE                           #
#===========================================================#
def magnitude_variance_pruning(masking, mask, weight, name):
    ''' Prunes weights which have high gradient variance and low magnitude.

    Weights with large magnitude and low variance are the most important.
    Weights with large magnitude but high variance hurt performance.
    Weights with small magnitude and low variance are useless.
    Weights with small magnitude and high variance cannot learn anything usefull.

    Weights with medium variance/magnitude are promising for improving network performance.

    Thus, we here take the geometric mean of those both distribution to find weights to prune.
    '''
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])

    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    max_var = torch.var(1./eight.grad[mask.byte()]).max().item()
    max_magnitude = torch.abs(weight.data[mask.byte()]).max().item()
    product = ((weight.grad/max_var)*torch.abs(weight.data)/max_magnitude)*mask
    product[mask==0] = 0.0

    x, idx = torch.sort(product.view(-1))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask


def your_growth(masking, name, new_mask, total_regrowth, weight):
    '''
    Returns:
        mask        Binary mask with newly grown weights.
                    1s indicated active weights in the binary mask.
    '''
    return new_mask


