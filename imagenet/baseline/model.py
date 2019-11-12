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
# Modifications copyright (C) 2019 Tim Dettmers
# CHANGES: 
#  - Replaced DynamicLinear layer with Linear layer in ResNet-50 to be compatible with sparselearning library.

import numpy as np
import math
import torch
import torch.nn as nn
from reparameterized_layers import DynamicLinear,DynamicConv2d
from parameterized_tensors import SparseTensor,TiedTensor
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

class DynamicNetworkBase(nn.Module):
    def __init__(self):
        super(DynamicNetworkBase, self).__init__()
        self.split_state = False

    def prune(self,prune_fraction_fc,prune_fraction_conv,prune_fraction_fc_special = None):
        for x in [x for x  in self.modules() if isinstance(x,SparseTensor)]:
            if x.conv_tensor:
                x.prune_small_connections(prune_fraction_conv)
            else:
                if x.s_tensor.size(0) == 10 and  x.s_tensor.size(1) == 100:
                    x.prune_small_connections(prune_fraction_fc_special)
                else:
                    x.prune_small_connections(prune_fraction_fc)
                    

                
    def get_model_size(self):
        def get_tensors_and_test(tensor_type):
            relevant_tensors = [x for x in self.modules() if isinstance(x,tensor_type)]
            relevant_params = [p for x in relevant_tensors for p in x.parameters()]
            is_relevant_param = lambda x : [y for y in relevant_params if x is y]

            return relevant_tensors,is_relevant_param

        sparse_tensors,is_sparse_param = get_tensors_and_test(SparseTensor)
        tied_tensors,is_tied_param = get_tensors_and_test(TiedTensor)

        
        sparse_params = [p for x in sparse_tensors for p in x.parameters()]
        is_sparse_param = lambda x : [y for y in sparse_params if x is y]


        sparse_size = sum([x.get_sparsity()[0].item() for x in sparse_tensors])

        tied_size = 0
        for k in tied_tensors:
            unique_reps = k.weight_alloc.cpu().unique()
            subtensor_size = np.prod(list(k.bank.size())[1:])
            tied_size += unique_reps.size(0) * subtensor_size
        
 
        fixed_size = sum([p.data.nelement()  for p in self.parameters() if (not is_sparse_param(p) and not is_tied_param(p))])
        model_size = {'sparse': sparse_size,'tied' : tied_size, 'fixed':fixed_size,'learnable':fixed_size + sparse_size + tied_size}    
        return model_size

    

class mnist_mlp(DynamicNetworkBase):

    def __init__(self,  initial_sparsity = 0.98,sparse = True,no_batch_norm = False):
        super(mnist_mlp, self).__init__()

        self.fc1 = DynamicLinear(784, 300, initial_sparsity,bias = no_batch_norm,sparse = sparse)
        self.fc_int = DynamicLinear(300, 100, initial_sparsity,bias = no_batch_norm,sparse = sparse)
        #self.fc2 = DynamicLinear(100, 10, min(0.5,initial_sparsity),bias = False,sparse = sparse)
        self.fc2 = DynamicLinear(100, 10, initial_sparsity,bias = no_batch_norm,sparse = sparse)

        if no_batch_norm:
            self.bn1 = lambda x : x
            self.bn2 = lambda x : x
            self.bn3 = lambda x : x
        else:
            self.bn1 = nn.BatchNorm1d(300)
            self.bn2 = nn.BatchNorm1d(100)
            self.bn3 = nn.BatchNorm1d(10)
            

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x.view(-1, 784))))
        x = F.relu(self.bn2(self.fc_int(x)))
        y = self.bn3(self.fc2(x))

        return y


    
#########Definition of wide resnets

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0,widen_factor = 10,initial_sparsity = 0.5,sub_kernel_granularity = False,sparse = True, sparse_momentum=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = DynamicConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                               padding=1, bias=False,initial_sparsity = initial_sparsity,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
            
        
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv2 = DynamicConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                               padding=1, bias=False,initial_sparsity = initial_sparsity,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
            
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        #self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               #padding=0, bias=False) or None
        self.convShortcut = (not self.equalInOut) and DynamicConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False, initial_sparsity=initial_sparsity, sub_kernel_granularity=sub_kernel_granularity, sparse=sparse) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0,widen_factor = 10,initial_sparsity = 0.5,sub_kernel_granularity = False,sparse = True):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,widen_factor,initial_sparsity = initial_sparsity,
                                      sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate,widen_factor,initial_sparsity = 0.5,sub_kernel_granularity = False,sparse = True):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,widen_factor = widen_factor,
                                initial_sparsity = initial_sparsity,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class cifar10_WideResNet(DynamicNetworkBase):
    def __init__(self, depth, num_classes=10, widen_factor=1, dropRate=0.0,initial_sparsity_conv = 0.5,initial_sparsity_fc = 0.95,sub_kernel_granularity = 4,sparse = True):
        super(cifar10_WideResNet, self).__init__()
        nChannels = np.round(np.array([16, 16*widen_factor, 32*widen_factor, 64*widen_factor])).astype('int32')
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate,widen_factor = widen_factor,
                                   initial_sparsity = initial_sparsity_conv,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate,widen_factor = widen_factor,
                                   initial_sparsity = initial_sparsity_conv,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate,widen_factor = widen_factor,
                                   initial_sparsity = initial_sparsity_conv,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)                                   

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3],num_classes) #DynamicLinear(nChannels[3], num_classes,initial_sparsity = initial_sparsity_fc,sparse = sparse)
        self.nChannels = nChannels[3]
        self.split_state = False
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, DynamicConv2d):
                n = m.kernel_size * m.kernel_size * m.n_output_maps
                if m.sparse:
                    m.d_tensor.s_tensor.data.normal_(0, math.sqrt(2. / n))
                else:
                    m.d_tensor.bank.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    

###Resnet Definition
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,widen_factor = 1,vanilla_conv1 = True,vanilla_conv3 = True,initial_sparsity = 0.5,
                 sub_kernel_granularity = 4,sparse = True):
        super(Bottleneck, self).__init__()
        adjusted_planes = planes#np.round(widen_factor * planes).astype('int32')
        
        #if vanilla_conv1:
        if not sparse:
            self.conv1 = nn.Conv2d(inplanes, adjusted_planes, kernel_size=1, bias=False)
            self.conv3 = nn.Conv2d(adjusted_planes, planes * 4, kernel_size=1, bias=False)
        else:
            self.conv1 = DynamicConv2d(inplanes, adjusted_planes, kernel_size=1, bias=False , initial_sparsity = initial_sparsity,
                                       sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
            self.conv3 = DynamicConv2d(adjusted_planes, planes * 4, kernel_size=1, bias=False , initial_sparsity = initial_sparsity,
                                       sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
        if not sparse:
            self.conv2 = nn.Conv2d(adjusted_planes, adjusted_planes, kernel_size=3, stride=stride,padding=1, bias=False)
        else:
            self.conv2 = DynamicConv2d(adjusted_planes, adjusted_planes, kernel_size=3, stride=stride,
                                           padding=1, bias=False,initial_sparsity = initial_sparsity, sub_kernel_granularity = sub_kernel_granularity,sparse = sparse)
            
            
        self.bn1 = nn.BatchNorm2d(adjusted_planes)
        self.bn2 = nn.BatchNorm2d(adjusted_planes)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(DynamicNetworkBase):

    def __init__(self, block, layers, num_classes=1000,widen_factor = 1,vanilla_downsample = False,vanilla_conv1 = False,vanilla_conv3 = False,
                 initial_sparsity_conv = 0.5,initial_sparsity_fc = 0.95,sub_kernel_granularity = 4,sparse = True):
        self.inplanes = np.round(64 * widen_factor).astype('int32')
        super(ResNet, self).__init__()
        self.widen_factor = widen_factor
        self.vanilla_conv1 = vanilla_conv1
        self.vanilla_conv3 = vanilla_conv3
        self.vanilla_downsample = vanilla_downsample        
        self.initial_sparsity_conv = initial_sparsity_conv
        self.initial_sparsity_fc = initial_sparsity_fc        
        self.sub_kernel_granularity = sub_kernel_granularity
        self.sparse = sparse
        
        #if not sparse:
        self.conv1 = nn.Conv2d(3, np.round(64 * widen_factor).astype('int32'), kernel_size=7, stride=2, padding=3,
                               bias=False)
        #else:
        #    self.conv1 = DynamicConv2d(3, np.round(64 * widen_factor).astype('int32'), kernel_size=7, stride=2, padding=3,
        #                           bias=False, initial_sparsity=initial_sparsity_conv, sub_kernel_granularity=sub_kernel_granularity, sparse=sparse)
        self.bn1 = nn.BatchNorm2d(np.round(64 * widen_factor).astype('int32'))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, np.round(64 * widen_factor).astype('int32'), layers[0])
        self.layer2 = self._make_layer(block, np.round(64 * widen_factor).astype('int32')*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, np.round(64 * widen_factor).astype('int32')*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, np.round(64 * widen_factor).astype('int32')*8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if not sparse:
            self.fc = nn.Linear(np.round(64 * widen_factor).astype('int32') * block.expansion * 8, num_classes,bias=True)
        else:
            self.fc = DynamicLinear(np.round(64 * widen_factor).astype('int32') * block.expansion * 8, num_classes,initial_sparsity = self.initial_sparsity_fc,sparse = sparse)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if sparse:
                    raise Exception('Used sparse=True, but some layers are still dense.')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, DynamicConv2d):
                if not sparse:
                    raise Exception('Used sparse=False, but some layers are still sparse.')
                n = m.kernel_size * m.kernel_size * m.n_output_maps
                if m.sparse:
                    m.d_tensor.s_tensor.data.normal_(0, math.sqrt(2. / n))
                else:
                    m.d_tensor.bank.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #if not self.sparse:
            conv = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            #else:
            #    DynamicConv2d(self.inplanes, planes * block.expansion,kernel_size=1,stride=stride, bias=False,
            #                          initial_sparsity = self.initial_sparsity_conv,sub_kernel_granularity = self.sub_kernel_granularity,sparse = self.sparse)
            downsample = nn.Sequential(conv, nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,widen_factor = self.widen_factor,
                            vanilla_conv1 = self.vanilla_conv1,vanilla_conv3 = self.vanilla_conv3,initial_sparsity = self.initial_sparsity_conv,
                            sub_kernel_granularity = self.sub_kernel_granularity,sparse = self.sparse))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,widen_factor = self.widen_factor,
                                vanilla_conv1 = self.vanilla_conv1,vanilla_conv3 = self.vanilla_conv3,initial_sparsity = self.initial_sparsity_conv,
                                sub_kernel_granularity = self.sub_kernel_granularity,sparse = self.sparse))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def imagenet_resnet50(widen_factor = 1,vanilla_conv1 = False,vanilla_conv3 = False,vanilla_downsample = True,decimation_factor = 8,
                      initial_sparsity_conv = 0.5,initial_sparsity_fc = 0.95,sub_kernel_granularity = 4,sparse = True, **kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],widen_factor = widen_factor,
                   vanilla_conv1 = vanilla_conv1,vanilla_conv3 = vanilla_conv3,vanilla_downsample = vanilla_downsample, initial_sparsity_conv = initial_sparsity_conv,
                   initial_sparsity_fc = initial_sparsity_fc,sub_kernel_granularity = sub_kernel_granularity,sparse = sparse,**kwargs)
    return model
