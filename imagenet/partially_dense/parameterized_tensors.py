# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter



def uniform_coverage(rank,n_features):
    reps = torch.zeros(n_features)
    place_element = torch.arange(rank)
    for i in np.arange(0,n_features,rank):
        reps[i:i+rank] = place_element[0:min(rank,n_features - i)]
    return reps.long()


class TiedTensor(nn.Module):

    def __init__(self, full_tensor_size,initial_sparsity,   sub_kernel_granularity = False):

        super(TiedTensor, self).__init__()

        ndim = len(full_tensor_size)
        assert ndim == 2 or ndim == 4, 'only 2D or 4D tensors supported'

        self.full_tensor_size = torch.Size(full_tensor_size)
        self.sub_kernel_granularity = sub_kernel_granularity
        
        n_alloc_elements =  np.prod(self.full_tensor_size).item() if sub_kernel_granularity else np.prod(self.full_tensor_size[:2]).item()

        self.num_weights = round((1 - initial_sparsity)*n_alloc_elements)
        
        
        self.register_buffer('weight_alloc',torch.zeros(n_alloc_elements).long())
        indices = np.arange(n_alloc_elements)
        np.random.shuffle(indices)
        self.weight_alloc[indices] = uniform_coverage(self.num_weights,n_alloc_elements)
 
        self.conv_tensor = False if ndim ==2 else True
       

        trailing_dimensions = [] if sub_kernel_granularity else self.full_tensor_size[2:]
        self.bank = Parameter(torch.Tensor(self.num_weights,*trailing_dimensions))

        self.init_parameters()
        
        
    def init_parameters(self):
        stdv = 1 / math.sqrt(np.prod(self.full_tensor_size[1:]))

        self.bank.data.uniform_(-stdv, stdv)
        self.bank.data[0] = 0.0
    def extra_repr(self):
        
        return 'full tensor size={} , unique_active_weights={}, fraction_of_total_weights = {}, sub_kernel_granularity = {}'.format(
            self.full_tensor_size, self.num_weights,self.num_weights * 1.0 / self.weight_alloc.size(0),self.sub_kernel_granularity)
        
    def forward(self):
        return self.bank[self.weight_alloc].view(self.full_tensor_size)


class SparseTensor(nn.Module):
    def __init__(self,tensor_size,initial_sparsity,sub_kernel_granularity = 4):
        super(SparseTensor,self).__init__()
        self.s_tensor = Parameter(torch.Tensor(torch.Size(tensor_size)))
        self.initial_sparsity = initial_sparsity
        self.sub_kernel_granularity = sub_kernel_granularity

        
        assert self.s_tensor.dim() == 2 or self.s_tensor.dim() == 4, "can only do 2D or 4D sparse tensors"


        trailing_dimensions = [1]*(4 - sub_kernel_granularity)
        self.register_buffer('mask',torch.Tensor(*(tensor_size[:sub_kernel_granularity] )))
        
        self.normalize_coeff = np.prod(tensor_size[sub_kernel_granularity:]).item()


        self.conv_tensor = False if self.s_tensor.dim() ==2 else True
        
        self.mask.zero_()
        flat_mask = self.mask.view(-1)
        indices = np.arange(flat_mask.size(0))
        np.random.shuffle(indices)
        flat_mask[indices[:int((1-initial_sparsity) * flat_mask.size(0) + 0.1)]] = 1

        self.grown_indices = None
        self.init_parameters()
        self.reinitialize_unused()

        self.tensor_sign = torch.sign(self.s_tensor.data.view(-1))

        
    def reinitialize_unused(self,reinitialize_unused_to_zero = True):
        unused_positions = (self.mask  < 0.5)
        if reinitialize_unused_to_zero:
            self.s_tensor.data[unused_positions] = torch.zeros(self.s_tensor.data[unused_positions].size()).to(self.s_tensor.device)
        else:
            if self.conv_tensor:
                n = self.s_tensor.size(0) * self.s_tensor.size(2) * self.s_tensor.size(3)
                self.s_tensor.data[unused_positions] = torch.zeros(self.s_tensor.data[unused_positions].size()).normal_(0, math.sqrt(2. / n)).to(self.s_tensor.device)                
            else:
                stdv = 1. / math.sqrt(self.s_tensor.size(1))            
                self.s_tensor.data[unused_positions] = torch.zeros(self.s_tensor.data[unused_positions].size()).normal_(0, stdv).to(self.s_tensor.device)                                
        
    def init_parameters(self):
        stdv = 1 / math.sqrt(np.prod(self.s_tensor.size()[1:]))

        self.s_tensor.data.uniform_(-stdv, stdv)

    def prune_sign_change(self,reinitialize_unused_to_zero = True,enable_print = False):
        W_flat = self.s_tensor.data.view(-1)

        new_tensor_sign = torch.sign(W_flat)
        mask_flat = self.mask.view(-1)        
        
        mask_indices = torch.nonzero(mask_flat > 0.5).view(-1)
        
        sign_change_indices = mask_indices[((new_tensor_sign[mask_indices] * self.tensor_sign[mask_indices].to(new_tensor_sign.device)) < -0.5).nonzero().view(-1)]
        
        mask_flat[sign_change_indices] = 0
        self.reinitialize_unused(reinitialize_unused_to_zero)

        cutoff = sign_change_indices.numel()
        
        if enable_print:
            print('pruned {}  connections'.format(cutoff))
        if self.grown_indices is not None and enable_print:
            overlap = np.intersect1d(sign_change_indices.cpu().numpy(),self.grown_indices.cpu().numpy())
            print('pruned {} ({} %) just grown weights'.format(overlap.size,overlap.size * 100.0 / self.grown_indices.size(0) if self.grown_indices.size(0) > 0  else 0.0))
        
        self.tensor_sign = new_tensor_sign
        return sign_change_indices
        
        
    def prune_small_connections(self,prune_fraction,reinitialize_unused_to_zero = True):
        if self.conv_tensor and self.sub_kernel_granularity < 4:
            W_flat = self.s_tensor.abs().sum(list(np.arange(self.sub_kernel_granularity,4))).view(-1) / self.normalize_coeff
        else:
            W_flat = self.s_tensor.data.view(-1)

        mask_flat = self.mask.view(-1)        
        
        mask_indices = torch.nonzero(mask_flat > 0.5).view(-1)

        
        W_masked = W_flat[mask_indices]
        
        sorted_W_indices = torch.sort(torch.abs(W_masked))[1]


        cutoff = int(prune_fraction * W_masked.numel()) + 1

        mask_flat[mask_indices[sorted_W_indices[:cutoff]]] = 0
        self.reinitialize_unused(reinitialize_unused_to_zero)

#        print('pruned {}  connections'.format(cutoff))
#        if self.grown_indices is not None:
#            overlap = np.intersect1d(mask_indices[sorted_W_indices[:cutoff]].cpu().numpy(),self.grown_indices.cpu().numpy())
            #print('pruned {} ({} %) just grown weights'.format(overlap.size,overlap.size * 100.0 / self.grown_indices.size(0)))
        
        return mask_indices[sorted_W_indices[:cutoff]]

    def prune_threshold(self,threshold,reinitialize_unused_to_zero = True):
        if self.conv_tensor and self.sub_kernel_granularity < 4:
            W_flat = self.s_tensor.abs().sum(list(np.arange(self.sub_kernel_granularity,4))).view(-1) / self.normalize_coeff
        else:
            W_flat = self.s_tensor.data.view(-1)

        mask_flat = self.mask.view(-1)        
        
        mask_indices = torch.nonzero(mask_flat > 0.5).view(-1)

        W_masked = W_flat[mask_indices]

        prune_indices = (W_masked.abs() < threshold).nonzero().view(-1)


        if mask_indices.size(0) == prune_indices.size(0):
            print('removing all. keeping one')
            prune_indices = prune_indices[1:]
        

        mask_flat[mask_indices[prune_indices]] = 0
        
 #       if mask_indices.numel() > 0 :
 #           print('pruned {}/{}({:.2f})  connections'.format(prune_indices.numel(),mask_indices.numel(),prune_indices.numel()/mask_indices.numel()))

#        if self.grown_indices is not None and self.grown_indices.size(0) != 0 :
#            overlap = np.intersect1d(mask_indices[prune_indices].cpu().numpy(),self.grown_indices.cpu().numpy())
#            print('pruned {} ({} %) just grown weights'.format(overlap.size,overlap.size * 100.0 / self.grown_indices.size(0)))

        self.reinitialize_unused(reinitialize_unused_to_zero)


        return mask_indices[prune_indices]
    
    def grow_random(self,grow_fraction,pruned_indices = None,enable_print = False,n_to_add = None):
        mask_flat = self.mask.view(-1)        
        mask_zero_indices = torch.nonzero(mask_flat < 0.5).view(-1)        
        if pruned_indices is not None:
            cutoff = pruned_indices.size(0)
            mask_zero_indices = torch.Tensor(np.setdiff1d(mask_zero_indices.cpu().numpy(),pruned_indices.cpu().numpy())).long().to(mask_zero_indices.device)
        else:
            cutoff = int(grow_fraction * mask_zero_indices.size(0))

        if n_to_add is not None:
           cutoff = n_to_add
        
            
        if mask_zero_indices.numel() < cutoff:
           print('******no place to grow {} connections, growing {} instead'.format(cutoff,mask_zero_indices.numel()))
           cutoff = mask_zero_indices.numel()

        if enable_print:
            print('grown {}  connections'.format(cutoff))        

        self.grown_indices = mask_zero_indices[torch.randperm(mask_zero_indices.numel())][:cutoff]
        mask_flat[self.grown_indices] = 1

        return cutoff

    def get_sparsity(self):
        active_elements = self.mask.sum() * np.prod(self.s_tensor.size()[self.sub_kernel_granularity:]).item() 
        return (active_elements,1 - active_elements / self.s_tensor.numel())


    def forward(self):
        if self.conv_tensor:
            return self.mask.view(*(self.mask.size() + (1,)*(4 - self.sub_kernel_granularity))) * self.s_tensor
        else:
            return self.mask * self.s_tensor
        

    def extra_repr(self):
        return 'full tensor size : {} , sparsity mask : {} , sub kernel granularity : {}'.format(
            self.s_tensor.size(), self.get_sparsity(),self.sub_kernel_granularity)
    

