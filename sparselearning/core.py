from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import math
import os
import shutil
import time
import types
from itertools import chain
import matplotlib
from matplotlib import pyplot as plt

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, gradient_neuron, covariance, variance.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, covariance, threshold.')
    parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: momentum, covariance, variance, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--sparse', action='store_true', default=True, help='Enable sparse mode. Default: True.')

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate



class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='random', redistribution_mode='variance', threshold=0.001):
        growth_modes = ['variance', 'random', 'covariance', 'momentum']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        print(growth_mode, death_mode, redistribution_mode)

        self.masks = {}
        self.modules = []
        self.grad_variance = {}
        self.names = []
        self.optimizer = optimizer
        self.variance_over_time = {}

        self.activation_variance = {}
        self.activation_variance_over_time = {}

        self.covariance_activation = {}
        self.grad_covariance = {}
        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}
        self.layer_norms = []

        self.grad_var = {}
        self.input_var = {}
        self.mask_over_time = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.excluded_layers = {}
        self.death_rate = death_rate
        self.name2death_rate = {}
        self.initialize = None
        self.steps = 0
        self.first_layer = None
        self.base_lr = None
        self.density_counter = {}
        self.name_to_32bit = {}
        self.half = False
        self.threshold = threshold
        self.growth_threshold = threshold
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        self.prune_every_k_steps = None
        self.global_init = False

    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True


    def init(self, mode='enforce_density_per_layer', density=0.05):
        self.initialize = mode
        self.sparsity = density
        self.init_optimizer()
        if mode == 'enforce_density_per_layer':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()
        elif mode == 'size_proportional':
            # initialization used in sparse evolutionary training
            total_params = 0
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    total_params += weight.numel()
                    self.baseline_nonzero += weight.numel()*density

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name][:] = (torch.rand(weight.shape) < prob).float().data.cuda()
            self.apply_mask()
        elif mode == 'passthrough':

            layer_number = 0
            nonzeros = []
            for name, module in list(self.modules[0].named_modules())[::-1]:
                if isinstance(module, nn.Linear):
                    module.bias.data *= 0.0
                    module.weight.data *= 0.0
                    in_dim = module.weight.shape[1]
                    out_dim = module.weight.shape[0]
                    neurons_per_feature = in_dim // out_dim
                    #if layer_number == 0: 
                    #    module.weight.data[:] = -1.0

                    j = 0
                    for i in range(out_dim):
                        for k in range(neurons_per_feature):
                            module.weight.data[i, j] = 1.0
                            j += 1
                    nonzeros.append(j)
                    self.first_layer = (module, name)
                    layer_number += 1
                    self.masks[name+'.weight'] = (module.weight.data != 0.0).float().cuda()
            # set first weight to zero

                if isinstance(module, nn.Conv2d):
                    raise NotImplemented('TODO: Conv2d')
            module, name = self.first_layer
            w = module.weight
            total_size = 0
            for weight in self.masks.values():
                total_size += weight.numel()
            self.baseline_nonzero = total_size*density
            module.weight.data *= 0.0
            budget = self.baseline_nonzero - sum(nonzeros[:-1])
            i = 0
            offset = 0
            while True:
                offset += 1
                for i in range(weight.shape[1]):
                    if budget == 0: break
                    k += offset
                    if k >= weight.shape[0]: k = 0
                    w.data[k, i] = 1.0
                    budget -= 1
                if budget == 0: break
            print(w.data.sum(1), w.data.sum(1).shape)
            print((w.data!= 0.0).sum().item(), (module.weight.data != 0.0).sum().item())

            self.masks[name+'.weight'] = (module.weight.data != 0.0).float().cuda()
            print((self.masks[name+'.weight'] != 0.0).sum().item())

            total_size = 0
            for name, weight in self.masks.items():
                total_size += (weight.data!= 0.0).sum().item()

            print(total_size)
            print(nonzeros)

        self.init_death_rate(self.death_rate)
        self.print_nonzero_counts()

        total_size = 0
        for name, module in self.modules[0].named_modules():
            if hasattr(module, 'weight'):
                total_size += module.weight.numel()
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    total_size += module.bias.numel()
        print('Total Model parameters:', total_size)

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total parameters after removed layers:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, density*total_size))

    def init_death_rate(self, death_rate):
        for name in self.masks:
            self.name2death_rate[name] = death_rate

    def at_end_of_epoch(self):
        self.truncate_weights()
        self.print_nonzero_counts()
        self.clear_gradients()
        self.reset_momentum()

    def step(self):
        self.sample_gradient()
        self.covariance_update()
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        for name in self.masks:
            self.name2death_rate[name] = self.death_rate_decay.get_dr(self.name2death_rate[name])

        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights(partial_name=None)
                self.print_nonzero_counts()
                self.clear_gradients()
                self.reset_momentum()




    def add_module(self, module, density, sparse_init='enforce_density_per_layer'):
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
        self.remove_weight_partial_name('bias')
        self.remove_type(nn.BatchNorm2d)
        self.remove_type(nn.BatchNorm1d)
        self.remove_type(nn.PReLU)
        self.init(mode=sparse_init, density=density)


    def register_layer_norm(self, partial_name):
        for module in self.modules:
            for m1 in module.modules():
                for key, value in m1._modules.items():
                    if partial_name in key:
                        self.layer_norms.append(value)
        print('Added {0} layer norm layers.'.format(len(self.layer_norms)))

    def reset_layer_norm(self):
        for norm in self.layer_norms:
            norm.reset_parameters()

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if verbose:
                    print('Removing {0}...'.format(name))
                removed.add(name)
                self.masks.pop(name)
        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1


    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight_partial_name(name, verbose=True)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if self.half:
                        tensor.data = tensor.data*self.masks[name].half()
                        tensor2 = self.name_to_32bit[name]
                        tensor2.data = tensor2.data*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name]

    def truncate_weights(self, partial_name=None):
        self.gather_statistics(partial_name)
        name2regrowth = self.calc_growth_redistribution(partial_name)

        total_nonzero_new = 0
        total_removed = 0
        if self.death_mode == 'global_magnitude':
            total_removed = self.global_magnitude_death()
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if partial_name is not None:
                        if isinstance(partial_name, set):
                            if not any([p in name for p in partial_name]): continue
                        elif partial_name not in name: continue
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(weight, name)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(weight, name)
                    elif self.death_mode == 'covariance':
                        new_mask = self.covariance_death(mask, weight, name)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name)

                    total_removed += self.name2nonzeros[name] - new_mask.sum().item()
                    #self.masks.pop(name)
                    #self.masks[name] = new_mask
                    self.masks[name][:] = new_mask


        if self.growth_mode == 'global_momentum':
            total_nonzero_new = self.global_gradient_growth(total_removed + self.adjusted_growth)
        else:
            if self.death_mode == 'threshold':
                expected_killed = sum(name2regrowth.values())
                #print(expected_killed, total_removed, self.threshold)
                if total_removed < (1.0-self.tolerance)*expected_killed:
                    self.threshold *= 2.0
                elif total_removed > (1.0+self.tolerance) * expected_killed:
                    self.threshold *= 0.5

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if partial_name is not None:
                        if isinstance(partial_name, set):
                            if not any([p in name for p in partial_name]): continue
                        elif partial_name not in name: continue
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    if self.death_mode == 'threshold':
                        total_regrowth = math.floor((total_removed/float(expected_killed))*name2regrowth[name]*self.growth_death_ratio)
                    elif self.redistribution_mode == 'none':
                        if name not in self.name2baseline_nonzero:
                            self.name2baseline_nonzero[name] = self.name2nonzeros[name]
                        old = self.name2baseline_nonzero[name]
                        new = new_mask.sum().item()
                        #print(old, new)
                        total_regrowth = int(old-new)
                    elif self.death_mode == 'global_magnitude':
                        expected_removed = self.baseline_nonzero*self.name2death_rate[name]
                        expected_vs_actual = total_removed/expected_removed
                        total_regrowth = math.floor(expected_vs_actual*name2regrowth[name]*self.growth_death_ratio)
                    else:
                        total_regrowth = math.floor(name2regrowth[name]*self.growth_death_ratio)




                    # growth
                    if self.growth_mode == 'covariance':
                        new_mask = self.covariance_growth(new_mask, total_regrowth, name, weight)
                    elif self.growth_mode == 'variance':
                        new_mask = self.variance_growth(new_mask, total_regrowth, name)
                    elif self.growth_mode == 'random':
                        new_mask = self.random_growth(new_mask, total_regrowth)
                    elif self.growth_mode == 'momentum':
                        new_mask = self.gradient_growth(name, new_mask, total_regrowth, weight)
                    elif self.growth_mode == 'gradient_neuron':
                        new_mask = self.gradient_neuron_growth(name, new_mask, total_regrowth, weight)


                    new_nonzero = new_mask.sum().item()

                    # storing mask for mask statistics
                    if name not in self.mask_over_time:
                        self.mask_over_time[name] = self.masks[name]
                    else:
                        self.mask_over_time[name] = self.mask_over_time[name] + self.masks[name]

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()
        self.reset_layer_norm()

        print(self.total_nonzero, self.baseline_nonzero, self.adjusted_growth)
        if self.baseline_nonzero is None: self.baseline_nonzero = self.total_nonzero
        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (death-growth) residuals to adjust future growth
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)



        if self.total_nonzero > 0:
            print('old, new nonzero count:', self.total_nonzero, total_nonzero_new, self.adjusted_growth)

        # adjust death rate
        #expected_variance = 1./len(self.name2variance)
        #for name, var in self.name2variance.items():
            #self.name2death_rate[name] *= 0.99
            #if var > expected_variance:
                #self.name2death_rate[name] *= 0.98

            #if var < expected_variance:
                #self.name2death_rate[name] *= 1.02




    '''
                    REDISTRIBUTION
    '''

    def gather_statistics(self, partial_name):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                if partial_name is not None:
                    if isinstance(partial_name, set):
                        if not any([p in name for p in partial_name]): continue
                    elif partial_name not in name: continue
                mask = self.masks[name]
                if self.redistribution_mode == 'momentum':
                    if self.half:
                        tensor = self.name_to_32bit[name]
                    #adam_grad = self.optimizer.state[tensor]['exp_avg']
                    #if hasattr(self.optimizer, '_get_param_groups'):
                        #print('groups!')
                    #print(list(self.optimizer._get_state()[tensor]))
                    if 'exp_avg' in self.optimizer.state[tensor]:
                        adam_m1 = self.optimizer.state[tensor]['exp_avg']
                        adam_m2 = self.optimizer.state[tensor]['exp_avg_sq']
                        grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
                    elif 'momentum_buffer' in self.optimizer.state[tensor]:
                        grad = self.optimizer.state[tensor]['momentum_buffer']

                    #self.name2variance[name] = torch.abs(tensor.grad*tensor)[mask.byte()].mean().item()#/(V1val*V2val)
                    #self.name2variance[name] = torch.abs(grad[mask==0]).mean().item()#/(V1val*V2val)
                    self.name2variance[name] = torch.abs(grad[mask.byte()]).mean().item()#/(V1val*V2val)
                    #print(name, self.name2variance[name])

                elif self.redistribution_mode == 'covariance':
                    C, mean, n = self.grad_covariance[name]
                    g1, m1, n1 = self.grad_var[name]
                    V2, m1, n1 = self.input_var[name]
                    Cval = torch.abs(C)[mask.byte()].mean().item()
                    #V1val = torch.abs(V1)[V1!=0.0].mean().item()
                    #V2val = torch.abs(V2)[V2!=0.0].mean().item()
                    #self.name2variance[name] = torch.abs(C)[mask == 0.0].mean().item()
                    #self.name2variance[name] = torch.abs(C)[mask.byte()].mean().item()
                    self.name2variance[name] = Cval#/(V1val*V2val)
                elif self.redistribution_mode == 'variance':
                    M, mean, n = self.grad_variance[name]
                    self.name2variance[name] = torch.abs(M)[mask.byte()].mean().item()
                elif self.redistribution_mode == 'magnitude':
                    self.name2variance[name] = torch.abs(tensor)[mask.byte()].mean().item()
                elif self.redistribution_mode == 'nonzeros':
                    self.name2variance[name] = float((torch.abs(tensor) > self.threshold).sum().item())
                elif self.redistribution_mode == 'none':
                    self.name2variance[name] = 1.0
                else:
                    print('Unknown redistribution mode:{0}'.format(self.redistribution_mode))
                    raise Exception('Unknown redistribution mode!')

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                self.total_removed += num_remove
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

        #val = np.array(list(self.name2variance.values()))
        #idx = np.arange(val.size)
        #np.random.shuffle(idx)
        #for i, name in enumerate(self.name2variance):
        #    print(self.name2variance[name], val[idx[i]])
        #    self.name2variance[name] = val[idx[i]]



    def calc_growth_redistribution(self, partial_name):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0
        for name in self.name2variance:
            self.name2variance[name] /= self.total_variance

        #for name in list(self.name2variance.keys()):
        #    n = self.name2nonzeros[name] + self.name2zeros[name]
        #    density = self.name2nonzeros[name] / float(n)
        #    var = self.name2variance[name]
        #    if density > 0.85:
        #        if name not in self.density_counter: self.density_counter[name] = 0
        #        self.density_counter[name] += 1
        #        if self.density_counter[name] > 10:
        #            print('Excluding weight {0} due excessive density.\n'.format(name))
        #            self.baseline_nonzero -= int(n)
        #            self.masks.pop(name)
        #            self.name2variance.pop(name)
        #            total_var = sum(self.name2variance.values())
        #            for key in self.name2variance:
        #                self.name2variance[key] /= total_var


        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                if partial_name is not None:
                    if isinstance(partial_name, set):
                        if not any([p in name for p in partial_name]): continue
                    elif partial_name not in name: continue
                #death_rate = min(self.name2death_rate[name], max(0.05, (self.name2zeros[name]/float(self.masks[name].numel()))))
                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

                #if regrowth > max_regrowth:
                #    name2regrowth[name] = max_regrowth
                if regrowth > 0.99*max_regrowth:
                    name2regrowth[name] = 0.99*max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0: mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        #nonzero = []
        #max_regrowth = []
        #var = []
        #drate = []
        #names = []

        #for name in self.name2variance:
        #    names.append(name)
        #    nonzero.append(self.name2nonzeros[name])
        #    max_regrowth.append(self.name2nonzeros[name] + self.name2zeros[name])
        #    var.append(self.name2variance[name])
        #    drate.append(self.name2death_rate[name])


        #nonzero = np.array(nonzero)
        #max_regrowth = np.array(max_regrowth)
        #var = np.array(var)
        #drate = np.array(drate)

        #total_removed = (drate*nonzero).sum()
        #expected_var = 1.0/var.size

        #total_regrowth = (var>expected_var)*drate*nonzero
        #total_removed -= total_regrowth.sum()

        #redistribute = (total_removed*var) + total_regrowth

        #name2regrowth = {}
        #for name, regrowth in zip(names, redistribute):
        #    name2regrowth[name] = regrowth

        return name2regrowth


    '''
                    DEATH
    '''
    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def covariance_death(self, mask, weight, name):
        num_remove = math.ceil(fraction*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        C, mean, n = self.grad_covariance[name]
        x, idx = torch.sort(torch.abs(C[mask.byte()]))

        threshold = x[num_remove].item()

        return mask.byte() & (torch.abs(C) > threshold)

    def magnitude_death(self, weight, name):
        sparsity = self.name2zeros[name]/float(self.masks[name].numel())
        death_rate = self.name2death_rate[name]
        if sparsity < 0.2:
            expected_variance = 1.0/len(list(self.name2variance.keys()))
            actual_variance = self.name2variance[name]
            expected_vs_actual = expected_variance/actual_variance
            if expected_vs_actual < 1.0:
                death_rate = min(sparsity, death_rate)
                print(name, expected_variance, actual_variance, expected_vs_actual, death_rate)
        num_remove = math.ceil(death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]
        num_nonzero = n-num_zeros

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)

    def global_magnitude_death(self):
        death_rate = 0.0
        for name in self.name2death_rate:
            if name in self.masks:
                death_rate = self.name2death_rate[name]
        tokill = math.ceil(death_rate*self.baseline_nonzero)
        total_removed = 0
        prev_removed = 0
        while total_removed < tokill*(1.0-self.tolerance) or (total_removed > tokill*(1.0+self.tolerance)):
            total_removed = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    remain = (torch.abs(weight.data) > self.threshold).sum().item()
                    total_removed += self.name2nonzeros[name] - remain

            if prev_removed == total_removed: break
            prev_removed = total_removed
            if total_removed > tokill*(1.0+self.tolerance):
                self.threshold *= 1.0-self.increment
                self.increment *= 0.99
            elif total_removed < tokill*(1.0-self.tolerance):
                self.threshold *= 1.0+self.increment
                self.increment *= 0.99

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name][:] = torch.abs(weight.data) > self.threshold

        return int(total_removed)


    def global_gradient_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    if 'exp_avg' in self.optimizer.state[weight]:
                        adam_m1 = self.optimizer.state[weight]['exp_avg']
                        adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
                        grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
                    elif 'momentum_buffer' in self.optimizer.state[weight]:
                        grad = self.optimizer.state[weight]['momentum_buffer']

                    grad = grad*(new_mask==0).float()
                    possible = (grad !=0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown


            if total_grown > togrow*(1.0+self.tolerance):
                self.growth_threshold *= 1.02
                #self.growth_increment *= 0.95
            elif total_grown < togrow*(1.0-self.tolerance):
                self.growth_threshold *= 0.98
                #self.growth_increment *= 0.95

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                if 'exp_avg' in self.optimizer.state[weight]:
                    adam_m1 = self.optimizer.state[weight]['exp_avg']
                    adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
                    grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
                elif 'momentum_buffer' in self.optimizer.state[weight]:
                    grad = self.optimizer.state[weight]['momentum_buffer']

                grad = grad*(new_mask==0).float()
                self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_death(self, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data > threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def covariance_growth(self, new_mask, total_regrowth, name, weight):
        C, mean, n = self.grad_covariance[name]
        C = C.data
        C = C*(new_mask==0).float()

        y, idx = torch.sort(torch.abs((C).flatten()))
        cov_threshold = y[-(total_regrowth-1)].item()
        new_mask = new_mask | (torch.abs(C) > cov_threshold)

        return new_mask

    def variance_growth(self, new_mask, total_regrowth, name):
        M, mean0, n = self.grad_variance[name]
        M = M.cuda()
        num_inputs = M.shape[1]

        nonzero_counts = []
        v = torch.zeros(M.shape[0], requires_grad=False).cuda()
        for i, m in enumerate(M):
            nonzero_mask = m>0
            nonzero_counts.append(nonzero_mask.sum().item())
            v[i] = m[nonzero_mask].sum().item()
        v.data /= v.data.sum()

        for i, fraction  in enumerate(v):
            neuron_regrowth = fraction*total_regrowth
            prob = (neuron_regrowth / (num_inputs-nonzero_counts[i]))
            new_mask[i] = new_mask[i] | (torch.rand(M.shape[1]).cuda() < prob)

        return new_mask

    def random_growth(self, new_mask, total_regrowth):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        return new_mask.byte() | new_weights

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if self.half:
            weight = self.name_to_32bit[name]

        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        grad = grad*(new_mask==0).float()
        #y, idx = torch.sort(torch.abs(grad[new_mask==0]).flatten())
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        threshold = y[total_regrowth].item()
        #threshold = y[-(total_regrowth-1)].item()
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
        #new_mask = new_mask | (torch.abs(grad) > threshold)

        return new_mask


    def gradient_neuron_growth(self, name, new_mask, total_regrowth, weight):
        if self.half:
            weight = self.name_to_32bit[name]

        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                if name in self.name2variance:
                    val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), self.name2variance[name])
                    print(val)
                else:
                    print(name, num_nonzeros)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.name2death_rate[name]))
                break

    def sample_gradient(self):
        if not (self.redistribution_mode == 'variance' or self.growth_mode == 'variance'): return
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                if name not in self.grad_variance: self.grad_variance[name] = [None, (tensor.grad.data*mask), 1]
                else:
                    grad = (tensor.grad.data*mask)
                    M, mean0, n = self.grad_variance[name]
                    n += 1
                    mean1 = mean0 + ((grad - mean0)*(1./n))
                    if n == 2:
                        M = (grad-mean0)*(grad-mean1)
                        self.grad_variance[name] = [M, mean1, n]
                    else:
                        M = M + (grad-mean0)*(grad-mean1)
                        self.grad_variance[name] = [M, mean1, n]

    def clear_gradients(self):
        for name in list(self.grad_variance.keys()):
            self.grad_variance.pop(name)

        for name in list(self.grad_covariance.keys()):
            self.grad_covariance.pop(name)

        for name in list(self.input_var.keys()):
            self.input_var.pop(name)

        for name in list(self.grad_var.keys()):
            self.grad_var.pop(name)

    def plot_variance_over_time(self, limit=50):
        if os.path.exists('./plots'):
            shutil.rmtree('./plots')
        os.mkdir('./plots/')

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.variance_over_time: continue
                values = self.variance_over_time[name]
                mask = values[0] > 0
                cols, rows = np.where(mask)
                nonzero = mask.sum()
                weights = tensor[mask].data.cpu().numpy()
                masked_values = []
                for v in values:
                    masked_values.append(v[mask].numpy())
                for i in range(nonzero):
                    weight_value = weights[i]
                    y = []
                    x = np.arange(len(values))
                    for v in masked_values:
                        y.append(v[i])

                    plt.plot(x, y)
                    plt.ylim((0, 0.0001))
                    plt.title('weight value: ' + str(weight_value) + ' at ' + str(rows[i]) + ',' + str(cols[i]))
                    #if i > 0 and i % 10 == 0:
                    plt.savefig('./plots/{0}_{2}_{1}_{3:.3f}.png'.format(name, i, cols[i], weight_value))
                    plt.clf()
                    if i >= limit: break

    def sample_activation_grad(self, name, x, grad):
        if name not in self.activation_variance: self.activation_variance[name] = [[], []]
        self.activation_variance[name][0].append(torch.mean(torch.abs(x.data),dim=0).cpu().numpy())
        self.activation_variance[name][1].append(torch.var(grad.data,dim=0).cpu().numpy())

    def clear_activation_grads(self):
        for name in list(self.activation_variance.keys()):
            if name not in self.activation_variance_over_time: self.activation_variance_over_time[name] = [[],[]]

            mean = self.activation_variance[name][0]
            var = self.activation_variance[name][1]
            n = len(mean)

            for i in range(n):
                if i == 0:
                    agg_mean = mean[0]
                    agg_var = var[0]
                else:
                    agg_mean += mean[i]
                    agg_var += var[i]

            agg_mean /= n
            agg_var /= n
            self.activation_variance_over_time[name][0].append(agg_mean)
            self.activation_variance_over_time[name][1].append(agg_var)
            self.activation_variance.pop(name)

    def add_covariance_activation(self, name, x):
        self.covariance_activation[name] = x

    def covariance_update(self):
        if not (self.redistribution_mode == 'covariance' or self.growth_mode == 'covariance'): return
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.covariance_activation: continue
                if name not in self.masks: continue
                a = self.covariance_activation[name].data
                # mean of activation over batch dimension [b, i] ->  [i]
                # mean of gradient over each neuron [h, i] -> [h]
                dims = list(range(len(a.shape)))[:-1]
                dims_tensor = list(range(len(tensor.shape))[::-1])[:-1]
                mask = self.masks[name]
                if name not in self.grad_covariance:
                    self.grad_covariance[name] = [None, a.mean(dims), 1]
                    self.grad_var[name] = [None, (tensor.grad.data*mask), 1]
                    self.input_var[name] = [None, a.mean(dims), 1]
                else:
                    activation = a.mean(dims)
                    raw_grad = tensor.grad.data*mask
                    grad = raw_grad.mean(dims_tensor)
                    #print(a.shape, raw_grad.shape)

                    # variance of gradient
                    #h_M_grad, h_mean0_grad, n = self.grad_var[name]
                    #mean0_grad = h_mean0_grad.cuda()
                    M_grad, mean0_grad, n = self.grad_var[name]

                    # variance of input
                    #h_M_input, h_mean0_input, xn = self.input_var[name]
                    #mean0_input = h_mean0_input.cuda()
                    M_input, mean0_input, xn = self.input_var[name]

                    n += 1
                    mean1_grad = mean0_grad + ((raw_grad - mean0_grad)*(1./n))
                    mean1_input = mean0_input + ((activation - mean0_input)*(1./n))

                    C, ymean0, yn = self.grad_covariance[name]
                    ymean1 = ymean0 + ((activation - ymean0)*(1./yn))

                    if n == 2:
                        M_grad = (raw_grad-mean0_grad)*(raw_grad-mean1_grad)
                        M_input = (activation-mean0_input)*(activation-mean1_input)
                        C = torch.ger(grad-mean0_grad.mean(dims_tensor),(activation-ymean1)).data.float()
                        #self.grad_var[name] = [M_grad.cpu(), mean1_grad.cpu(), n]
                        #self.input_var[name] = [M_input.cpu(), mean1_input.cpu(), n]
                        self.grad_covariance[name] = [C, ymean1, n]
                        self.grad_var[name] = [M_grad, mean1_grad, n]
                        self.input_var[name] = [M_input, mean1_input, n]
                    else:
                        #M_grad = h_M_grad.cuda()
                        #M_input = h_M_input.cuda()
                        #M_grad = h_M_grad.cuda()
                        #M_input = h_M_input.cuda()
                        #print('var', name, activation.std(), raw_grad[mask.byte()].std())
                        C = C + (torch.ger(grad-mean0_grad.mean(1),activation-ymean1).data.float())
                        M_grad = M_grad + (raw_grad-mean0_grad)*(raw_grad-mean1_grad)
                        M_input = M_input + (activation-mean0_input)*(activation-mean1_input)
                        self.grad_covariance[name] = [C, ymean1, n]
                        self.grad_var[name] = [M_grad, mean1_grad, n]
                        self.input_var[name] = [M_input, mean1_input, n]

    def plot_activation_variance(self, limit=500):
        if os.path.exists('./plots'):
            shutil.rmtree('./plots')
        os.mkdir('./plots/')

        for name, values in self.activation_variance_over_time.items():
            mean, var = values
            num_units = var[0].shape[0]
            for i in range(num_units):
                x = np.arange(len(mean))
                y = []
                for v in var:
                    y.append(v[i])

                plt.plot(x, y)
                y = []
                for v in mean:
                    y.append(v[i]/500000.)
                plt.plot(x, y)
                plt.ylim((0, 0.00001))
                plt.savefig('./plots/{0}_{1}.png'.format(name, i))
                plt.clf()
                if i >= limit: break

    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                    elif w == 'step_size':
                        #set to learning rate
                        self.optimizer.state[self.weight]['step_size'][indices] = optimizer.defaults['lr']
                    elif w == 'prev':
                        self.optimizer.state[self.weight]['prev'][indices] = 0.0
                    # TODO: Memory leak
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        #use average of all for very rough estimate of stable value
                        #self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w])
                        #self.optimizer.state[tensor][w] = self.optimizer.state[tensor][w]*mask
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                        #print(self.optimizer.state[tensor][w][mask==0].sum())
                    elif w == 'acc_delta':
                        #set to learning rate
                        self.optimizer.state[self.weight]['step_size'][indices] = 0.0
                    #elif buffer == ''

def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0


    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0: print(batch_idx,'/', len(test_loader))
        with torch.no_grad():
            #if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                #print('=='*50)
                #print('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                #print(len(feats))

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        #print(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        #print(feat_id, map_id, cls)
                        #print(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                model.feats = []

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        #print(feat_id, data)
        full_contribution = data.sum()
        #print(full_contribution, data)
        contribution_per_channel = ((1.0/full_contribution)*data.sum(1))
        #print('pre', data.shape[0])
        channels = data.shape[0]
        #data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print(data.shape, 'pre')
        data = data[idx[threshold_idx:]]
        print(data.shape, 'post')

        #perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        #print(contribution_per_channel, perc, feat_id)
        #data = data[contribution_per_channel > perc]
        #print(contribution_per_channel[contribution_per_channel < perc].sum())
        #print('post', data.shape[0])
        normed_data = np.max(data/np.sum(data,1).reshape(-1, 1), 1)
        #normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        #counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        sparse = False
        np.save('./results/alexnet_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense'), normed_data)
        #np.save('./results/VGG_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense'), normed_data)
        #np.save('./results/WRN-28-2_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense'), normed_data)
        #plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        #plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        #plt.xlim(0.1, 0.5)
        #if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        #else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        #plt.clf()

class DatasetSplitter(torch.utils.data.Dataset):
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_cifar10_dataloaders(args, validation_split=0.0):

    #normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                               #std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    #train_transform = transforms.Compose([
    #    transforms.RandomCrop(32, padding=4),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                         (0.2023, 0.1994, 0.2010)),
    #])

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                    (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.normalize((0.4914, 0.4822, 0.4465),
                             #(0.2023, 0.1994, 0.2010)),
         normalize
    ])

    full_dataset = datasets.CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = datasets.CIFAR10('_dataset', False, test_transform, download=False)



    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def get_mnist_dataloaders(args, validation_split=0.0):

    normalize = transforms.Normalize((0.13057,),(0.308011,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    #b = datasets.MNIST('../data', train=True, download=True)
    #x = np.float32(b.data)
    #x /= 255.

    #train = x[split:]
    #print(train.mean(), train.std())

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
