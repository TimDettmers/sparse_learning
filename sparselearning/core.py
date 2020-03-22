from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import math
import os
import shutil
import time
import bisect
from matplotlib import pyplot as plt
from sparselearning.funcs import redistribution_funcs, growth_funcs, prune_funcs
from functools import reduce

torch.set_printoptions(sci_mode=False)

class CUDATimer(object):
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def tick(self):
        self.start.record()

    def tock(self):
        self.end.record()
        self.start.synchronize()
        self.end.synchronize()
        return self.start.elapsed_time(self.end)/1000.0

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
    parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--dense', action='store_true', help='Enable dense mode. Default: False.')
    parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')

class AbstractLossEngine(object):
    def __init__(self):
        self.num_samples = 0

    def loss_func(self, inputs):
        raise NotImplementedError('Implement loss function!')

    def select_func(self, inputs, idx):
        raise NotImplementedError('Implement select function!')

    def make_batch(self):
        raise NotImplementedError('Implement make batch function!')

class CVLossEngine(AbstractLossEngine):
    def __init__(self, model):
        super(CVLossEngine, self).__init__()
        self.model = model
        self.x = []
        self.y = []

    def loss_func(self, inputs):
        with torch.no_grad():
            x, y = inputs
            output = self.model(x)
            loss = F.nll_loss(output, y, reduction='none')
        return loss


    def select_func(self, inputs, idx):
        x, y = inputs
        self.num_samples += idx.shape[0]
        self.x.append(x.data[idx].clone())
        self.y.append(y.data[idx].clone())

    def make_batch(self, batch_size):
        data = torch.cat(self.x, 0)
        target = torch.cat(self.y, 0)
        data = data[:batch_size]
        target = target[:batch_size]

        # cleanup
        self.num_samples = 0
        del self.x[:]
        del self.y[:]
        self.x = []
        self.y = []

        return [data, target]

class FairseqLossEngine(AbstractLossEngine):
    def __init__(self, model, criterion, method='sum'):
        super(FairseqLossEngine, self).__init__()
        self.sampled_data = {}
        self.model = model
        self.criterion = criterion
        self.bin_data = None
        self.bin_range = None
        self.iter = 0
        self.word2losshistory = {}
        self.n = 0
        self.freqtbl = torch.ones(1000000, dtype=torch.float32, device='cuda')
        self.warmup = 0
        self.n_looked_at = 0
        self.method = method

    def loss_func(self, inputs):
        n = inputs['nsentences']
        with torch.no_grad():
            loss, sample_size, logging_output = self.criterion(self.model, inputs, reduce=False)

        tokens = inputs['net_input']['src_tokens']
        idx, counts = torch.unique(tokens, return_counts=True)
        self.freqtbl[idx] += counts
        self.n += counts.sum().item()

        #for l, idx in zip(loss.reshape(n, -1), inputs['net_input']['src_tokens']):
        #    for word_loss, word_idx in zip(l, idx):
        #        if word_idx.item() not in self.word2losshistory: self.word2losshistory[word_idx.item()] = []
        #        self.word2losshistory[word_idx.item()].append(word_loss.item())
        #    counts = torch.histc(l, 25, 5, 15)
        #    self.iter += 1
        #    if self.bin_data is None:
        #        self.bin_data = counts
        #        self.bin_range = torch.arange(5, 15, (15-5)/25)
        #    else:
        #        self.bin_data += counts

        #    if self.iter > 0 and self.iter % 500 == 0:
        #        print('='*80)
        #        print(self.iter)
        #        print(self.bin_data)
        #        print(self.bin_range)
        #        max_val = 0
        #        #for key, values in self.word2losshistory.items():
        #        #    local_max = reduce(max, values)
        #        #    max_val = max(max_val, local_max)
        #        #    if local_max > max_val*0.75 and len(values) > 2:
        #        #        print(max_val, key, values)

        #        self.bin_data = None
        #        self.bin_range = None
        #        self.word2losshistory = {}

        rescaled_loss = loss.data.clone()
        if self.method == 'sum':
            return rescaled_loss.reshape(n, -1).sum(1)
        elif self.method == 'max':
            return rescaled_loss.reshape(n, -1).max(1)[0]
        elif self.method == 'rescaled_sum':
            rescaled_loss = rescaled_loss*self.freqtbl[tokens.view(-1)]/self.n
            return rescaled_loss.reshape(n, -1).sum(1)
        elif self.method == 'inverse_rescaled_sum':
            rescaled_loss = rescaled_loss/(self.freqtbl[tokens.view(-1)]/self.n)
            return rescaled_loss.reshape(n, -1).sum(1)
        elif self.method == 'rescaled_max':
            rescaled_loss = rescaled_loss*self.freqtbl[tokens.view(-1)]/self.n
            return rescaled_loss.reshape(n, -1).max(1)[0]
        elif self.method == 'inverse_rescaled_max':
            rescaled_loss = rescaled_loss/(self.freqtbl[tokens.view(-1)]/self.n)
            return rescaled_loss.reshape(n, -1).max(1)[0]
        elif self.method == 'percentile':
            #print(rescaled_loss.data.reshape(n, -1)[0])
            k = int(loss.reshape(n, -1).shape[1]*0.9)
            return torch.kthvalue(rescaled_loss.reshape(n, -1), k=k, dim=1)[0]
        elif self.method == 'topk_sum':
            k = int(loss.reshape(n, -1).shape[1]*0.1)
            maxval, maxidx = torch.topk(rescaled_loss.reshape(n, -1), k=k, dim=0)
            return maxval.sum(1)


        # get 90th percentile

        #return rescaled_loss.reshape(n, -1).max(1)[0]



    def select_func(self, inputs, idx):
        self.iter += 1
        if self.iter < self.warmup:
            idx = torch.arange(inputs['nsentences'],device=idx.device)

        n = idx.shape[0]
        for key, value in inputs.items():
            if isinstance(value, int):
                self.sampled_data[key] = value
                continue
            elif isinstance(value, dict):
                if key not in self.sampled_data: self.sampled_data[key] = {}
                for key2, value2 in value.items():
                    if key2 not in self.sampled_data[key]: self.sampled_data[key][key2] = []
                    self.sampled_data[key][key2].append(value2.data[idx].clone())
            else:
                if key not in self.sampled_data: self.sampled_data[key] = []
                self.sampled_data[key].append(value.data[idx].clone())
        self.num_samples += n
        self.n_looked_at += inputs['nsentences']

    def make_batch(self, batch_size):
        data = {}
        leftovers = {}
        leftover_size = 0
        for key, values in self.sampled_data.items():
            if isinstance(values, int):
                data[key] = values
            elif isinstance(values, dict):
                data[key] = {}
                leftovers[key] = {}
                for key2, values2 in values.items():
                    value2 = torch.cat(values2, 0)
                    if value2.shape[0] > batch_size:
                        leftovers[key][key2] = [value2[batch_size:]]
                        value2 = value2[:batch_size]
                    data[key][key2] = value2
            else:
                value = torch.cat(values, 0)
                if value.shape[0] > batch_size:
                    leftovers[key] = [value[batch_size:]]
                    leftover_size = value.shape[0] - batch_size
                    value = value[:batch_size]
                data[key] = value

        # cleanup
        self.num_samples = leftover_size
        self.n_looked_at = leftover_size
        leftovers['nsentences'] = leftover_size
        leftovers['ntokens'] = self.sampled_data['ntokens']
        del self.sampled_data
        self.sampled_data = leftovers

        return data


class SelectiveBackpropSampler(object):
    def __init__(self, loss_engine, beta, seed=0, max_size=1000, epsilon=0.5, decay=0.999):
        self.beta = beta
        self.history = []
        self.history2 = []
        self.rdm = np.random.RandomState(seed)
        self.max_size = max_size
        self.sampled_batch = []
        self.batch_size = None
        self.counts = [0,0]
        self.sampled_data = {}
        self.num_samples = 0
        self.t = CUDATimer()
        self.engine = loss_engine
        self.epsilon = epsilon
        self.decay = decay
        self.iter = 0
        self.clean_interval = 50

    def generate_sample(self, inputs, idx, batch_size):
        self.batch_size = batch_size
        loss = self.engine.loss_func(inputs)
        sampled_idx = self.histogram_sample(loss, idx)
        if sampled_idx.numel() > 0:
            self.engine.select_func(inputs, sampled_idx)
        self.counts[0] += 1

        if self.engine.num_samples < self.batch_size: return None


        self.counts[1] += 1

        self.epsilon *= self.decay
        self.iter += 1

        if self.counts[1] % 100 == 0:
            print('#Forward: {0}; #Backward: {1} Selectivity: {2:.4f}'.format(self.counts[0], self.counts[1], self.counts[1]/self.counts[0]))
            print('Epsilon: {0:.4f}'.format(self.epsilon))

        return self.engine.make_batch(self.batch_size)


    def histogram_sample(self, loss, indices):
        if self.rdm.rand(1) < self.epsilon: return torch.arange(0, loss.shape[0], device=loss.device)
        if len(self.history2) > self.max_size:
            #self.history2 = self.history2[-self.max_size:]
            del self.history2[:]
            self.history2 = []

        if self.iter % self.clean_interval and self.iter > 0:
            upper = loss.mean() + (loss.std()*3)
            lower = loss.mean() - (loss.std()*3)

            i = 0
            #print('pre', loss.mean(), torch.cat(self.history2).mean())
            while i < len(self.history2):
                m = self.history2[i].mean()
                if m < lower or m > upper:
                    self.history2.pop(i)
                else:
                    i += 1
            #print('post', loss.mean(), torch.cat(self.history2).mean())

        self.history2.append(loss.clone())

        data = torch.cat(self.history2).view(-1, 1).expand(-1, loss.shape[0])
        m1 = loss.mean()
        m2 = data.mean()
        data = data - (m2-m1)
        #percentiles = torch.pow((loss > data).sum(0).float()/data.shape[0], self.beta)
        #rdm = torch.rand(percentiles.shape[0], device=percentiles.device)
        percentiles = (loss > data).sum(0).float()/data.shape[0]


        #print(rdm.device, percentiles.device)
        #print(1.0-self.beta)
        #print(percentiles.mean().item())
        idx = torch.where(percentiles > (1.0 - self.beta))[0]


        #print(idx.numel()/ loss.numel(), self.beta)

        #print(idx)
        #print(idx)
        #selected = idx.cpu().numpy().tolist()

        return idx
        ##print(idx)


    def get_samples2(self, loss, indices):
        selected = []
        rdms = self.rdm.rand(loss.shape[0])
        for l, idx, rdm in zip(loss, indices, rdms):
            bisect.insort(self.history, l.item())

            length = len(self.history)
            pos = bisect.bisect(self.history, l.item())
            percentile = np.power(pos/length, self.beta)
            if rdm < percentile:
                selected.append(idx.item())

        if len(self.history) > self.max_size:
            self.history = []
            print('freeing history...')

        return selected





class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    """Anneals the pruning rate linearly with each step."""
    def __init__(self, prune_rate, T_max):
        self.steps = 0
        self.decrement = prune_rate/float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.steps += 1
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate



class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)

    Removing layers: Layers can be removed individually, by type, or by partial
    match of their name.
      - `mask.remove_weight(name)` requires an exact name of
    a parameter.
      - `mask.remove_weight_partial_name(partial_name=name)` removes all
        parameters that contain the partial name. For example 'conv' would remove all
        layers with 'conv' in their name.
      - `mask.remove_type(type)` removes all layers of a certain type. For example,
        mask.remove_type(torch.nn.BatchNorm2d) removes all 2D batch norm layers.
    """
    def __init__(self, optimizer, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', verbose=False, fp16=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose

        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}

        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        self.start_name = None

        # global growth/prune state
        self.prune_threshold = 0.001
        self.growth_threshold = 0.001
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        self.prune_every_k_steps = None
        self.half = fp16
        self.name_to_32bit = {}


    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True

    def init(self, mode='constant', density=0.05):
        self.sparsity = density
        self.init_growth_prune_and_redist()
        self.init_optimizer()
        if mode == 'constant':
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()
        elif mode == 'resume':
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    self.masks[name][:] = (weight != 0.0).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()
        elif mode == 'linear':
            # initialization used in sparse evolutionary training
            # scales the number of non-zero weights linearly proportional
            # to the product of all dimensions, that is input*output
            # for fully connected layers, and h*w*in_c*out_c for conv
            # layers.

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

    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')

    def at_end_of_epoch(self):
        self.truncate_weights()
        if self.verbose:
            self.print_nonzero_counts()

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)

        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                if self.verbose:
                    self.print_nonzero_counts()

    def add_module(self, module, density, sparse_init='constant'):
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d, verbose=self.verbose)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d, verbose=self.verbose)
        self.init(mode=sparse_init, density=density)

    def is_at_start_of_pruning(self, name):
        if self.start_name is None: self.start_name = name
        if name == self.start_name: return True
        else: return False

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name+'.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name+'.weight'].shape, self.masks[name+'.weight'].numel()))
            self.masks.pop(name+'.weight')
        else:
            print('ERROR',name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape, np.prod(self.masks[name].shape)))
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
                    self.remove_weight(name)
                    #self.remove_weight_partial_name(name, verbose=self.verbose)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name not in self.name2prune_rate: self.name2prune_rate[name] = self.prune_rate

                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                if sparsity < 0.2:
                    # determine if matrix is relativly dense but still growing
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        # growing
                        self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def truncate_weights(self):
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        if self.global_prune:
            self.total_removed = self.prune_func(self)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # prune
                    new_mask = self.prune_func(self, mask, weight, name)
                    removed = self.name2nonzeros[name] - new_mask.sum().item()
                    self.total_removed += removed
                    self.name2removed[name] = removed
                    self.masks[name][:] = new_mask

        name2regrowth = self.calc_growth_redistribution()
        if self.global_growth:
            total_nonzero_new = self.growth_func(self, self.total_removed + self.adjusted_growth)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.bool()

                    # growth
                    new_mask = self.growth_func(self, name, new_mask, math.floor(name2regrowth[name]), weight)
                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (prune-growth) residuals to adjust future growth
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        if self.total_nonzero > 0 and self.verbose:
            print('Nonzero before/after: {0}/{1}. Growth adjustment: {2:.2f}.'.format(
                  self.total_nonzero, total_nonzero_new, self.adjusted_growth))

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}
        self.name2removed = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                # redistribution
                self.name2variance[name] = self.redistribution_func(self, name, weight, mask)

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

        for name in self.name2variance:
            if self.total_variance != 0.0:
                self.name2variance[name] /= self.total_variance
            else:
                print('Total variance was zero!')
                print(self.growth_func)
                print(self.prune_func)
                print(self.redistribution_func)
                print(self.name2variance)

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                prune_rate = self.name2prune_rate[name]
                num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

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

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if self.prune_mode == 'global_magnitude':
                    expected_removed = self.baseline_nonzero*self.name2prune_rate[name]
                    if expected_removed == 0.0:
                        name2regrowth[name] = 0.0
                    else:
                        expected_vs_actual = self.total_removed/expected_removed
                        name2regrowth[name] = math.floor(expected_vs_actual*name2regrowth[name])

        return name2regrowth


    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

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

        print('Prune rate: {0}\n'.format(self.prune_rate))
