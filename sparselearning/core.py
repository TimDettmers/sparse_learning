from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

import numpy as np
import math
import os
import shutil
import time
import seaborn as sb
from matplotlib import pyplot as plt
from sparselearning.funcs import redistribution_funcs, growth_funcs, prune_funcs


def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
    parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--dense', action='store_true', help='Enable dense mode. Default: False.')
    parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(13*10, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.fc(x)


class CorrelationTracker(object):
    def __init__(self, num_labels, momentum=0.9):
        super(CorrelationTracker, self).__init__()
        self.label = None
        self.num_labels = num_labels
        self.name2corr = {}
        self.momentum = momentum
        self.label_vector = None
        self.name2train_accs = {}
        self.name2val_accs = {}
        self.val_accs_ensemble = []
        self.name2clusters = {}
        self.name2prev_clusters = {}
        self.prev_idx = None
        self.iter = 0
        self.name2module = {}
        self.name2idx = {}
        self.idx2name = {}
        self.name2prev_idx = {}
        self.train_votes = []
        self.val_votes = []
        self.lbls = []
        self.w = torch.nn.Linear(13*10, 10, bias=True).cuda()
        self.clf = LogisticRegression(solver='sag', multi_class='auto', max_iter=100)
        self.model = LinearModel().cuda()
        #self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.opt = torch.optim.SGD(self.w.parameters(), lr=0.01, momentum=0.9)

    def wrap_model(self, model):
        for n, m in model.named_modules():
            #if isinstance(m, torch.nn.ReLU):
            #    idx = self.name2idx[n]
            #    m1 = self.name2module[self.idx2name[idx-1]] if idx-1 > 0 else None
            #    m2 = self.name2module[self.idx2name[idx-2]] if idx-2 > 0 else None
            #    if (m1 is not None and isinstance(m1, torch.nn.Conv2d)) or \
            #       (m2 is not None and isinstance(m2, torch.nn.Conv2d)):
            #        m.register_forward_hook(lambda module, inputs, output, name=n: self.forward_conv(name, module, inputs, output))

            if isinstance(m, torch.nn.Conv2d):
                m.register_forward_hook(lambda module, inputs, output, name=n: self.forward_conv(name, module, inputs, output))
            m.register_forward_pre_hook(lambda module, inputs, name=n: self.forward_other(name, module, inputs))

    def network_class_correlation_plot(self, path):
        if not os.path.exists(path): os.makedirs(path)

        dmax = 0
        for name, corr in self.name2corr.items():
            if corr.shape[0] > dmax:
                dmax = corr.shape[0]


        network = np.zeros((dmax, len(self.name2corr)), dtype=np.float32)
        for cls in range(1):
            print('Plotting for class {0}...'.format(cls))
            names = list(self.name2corr.keys())
            layer_idx = 0
            print(names)
            for name1, name2 in zip(names[:-1], names[1:]):
                corr1 = self.name2corr[name1].clone().mean([1, 2])
                corr2 = self.name2corr[name2].clone().mean([1, 2])
                module2 = self.name2module[name2]
                w = module2.weight.data.sum([2, 3]).permute([1,0])
                n1 = corr1.shape[0]
                n2 = corr2.shape[0]
                #network[:n1, layer_idx] = corr1[:, cls].cpu().numpy()
                layer_idx += 1

                dmax_local = max(corr1.shape[0], corr2.shape[0])
                rep = 20
                l2l = np.zeros((n1+rep, n2+rep), dtype=np.float32)
                # corr1 [n1, 1]
                # corr2 [n2, 1]
                # l2l [n1+1, n2+1]
                l2l[:-rep, :rep] = corr1[:, cls].unsqueeze(-1).cpu().numpy()
                l2l[:-rep, rep:] = MinMaxScaler(feature_range=(-0.8, 0.8)).fit_transform(w.cpu().numpy())
                #l2l[:-rep, rep:] = StandardScaler().fit_transform(w.cpu().numpy())
                #l2l[:-rep, rep:]/= np.abs(l2l[:-rep, rep:]).max(0)/0.8
                l2l[-rep:, rep:] = corr2[:, cls].unsqueeze(0).cpu().numpy()
                fig = sb.heatmap(l2l, vmin=-0.8, vmax=0.8)
                #print(fig)

                #fig.set_size_inches(18.5, 10.5)
                plt.savefig(os.path.join(path, '{0}_to_{1}_{2}.png'.format(name1, name2,cls)), dpi=300)
                plt.close()
                plt.clf()

            #sb.heatmap(network, vmin=-0.8, vmax=0.8)
            #plt.savefig(os.path.join(path, '{0}.png'.format(cls)))
            #plt.clf()




    def propagate_correlations(self):
        #if len(self.name2prev_clusters) == 0: return
        names = list(self.name2corr.keys())
        for name1, name2 in zip(names[:-1], names[1:]):
            corr1 = self.name2corr[name1]
            corr2 = self.name2corr[name2]
            module2 = self.name2module[name2]
            #clusters1 = self.name2prev_clusters[name1]

            padded_corr1 = corr1.clone().permute([1, 0]).unsqueeze(-1).unsqueeze(-1)
            padded_corr1 = F.pad(padded_corr1, [1, 1, 1, 1])

            with torch.no_grad():
                prop_corr = F.conv2d(padded_corr1, module2.weight, module2.bias, module2.stride, module2.padding, module2.dilation)
            corr3 = prop_corr.mean([2, 3]).permute([1, 0])
            print(name2)
            print('Baseline: ', torch.sqrt((corr2-corr3)**2).mean(0))

            print(module2.weight.shape, corr1.shape, corr2.shape)

            corr4 = module2.weight.unsqueeze(-1)*corr1.view(1, corr1.shape[0], 1, 1, corr1.shape[1])
            print(corr4.shape)
            corr4 = corr4.sum([1, 2, 3])
            print(corr4.shape)
            print(corr2[:3])
            print(corr4[:3])
            print(F.normalize(corr2[:3]))
            print(F.normalize(corr4[:3]))
            print('Baseline: ', torch.sqrt((corr2-corr4)**2).mean(0))

            #labels = torch.eye(self.num_labels).to(corr1.device)
            #labels[labels == 0.0] = -1.0/self.num_labels
            #feats = torch.mm(labels, torch.mm(corr1.T, torch.mm(corr1, corr1.T)))
            #feats_sorted = torch.argsort(feats, dim=1, descending=True)

            #for k in range(corr1.shape[0]):
            #    partial_corr = torch.zeros_like(corr1)
            #    partial_corr.zero_()
            #    idx = feats_sorted[:, :k].reshape(-1)
            #    partial_corr[idx] = corr1[idx]

            #    padded_corr1 = partial_corr.permute([1, 0]).unsqueeze(-1).unsqueeze(-1)
            #    padded_corr1 = F.pad(padded_corr1, [1, 1, 1, 1])

            #    with torch.no_grad():
            #        prop_corr = F.conv2d(padded_corr1, module2.weight, module2.bias, module2.stride, module2.padding, module2.dilation)
            #    corr3 = prop_corr.mean([2, 3]).permute([1, 0])
            #    print('K-features {0}: {1}'.format(k, torch.sqrt((corr2-corr3)**2).mean(0)))



            #padded_corr2 = corr2.clone().permute([1, 0]).unsqueeze(-1).unsqueeze(-1)
            #padded_corr2 = F.pad(padded_corr2, [1, 1, 1, 1])
            #with torch.no_grad():
            #    prop_corr2 = F.conv_transpose2d(padded_corr2, module2.weight, None, module2.stride, module2.padding)
            #corr4 = prop_corr2.mean([2, 3]).permute([1, 0])
            #corr1 = self.name2corr[name1]
            #print(name1, torch.sqrt((corr1-corr4)**2).mean(0))

            print('='*80)

    def build_graph(self, model):
        i = 0
        for pname, pmodule in model.named_modules():
            num_children = len(list(pmodule.named_children()))
            if num_children > 0:
                for cname, cmodule in pmodule.named_children():
                    if pname == '': continue

                    name = '{0}.{1}'.format(pname, cname)
                    self.name2module[name] = cmodule
                    self.name2idx[name] = i
                    self.idx2name[i] = name
                    i+= 1
            else:
                if pname == '': continue
                name = '{0}'.format(pname)
                self.name2module[name] = pmodule
                self.name2idx[name] = i
                self.idx2name[i] = name
                i+= 1

        for i in range(len(self.idx2name)):
            name = self.idx2name[i]
            module = self.name2module[name]

    def set_label(self, label):
        # if mini-batch size is different or label not set
        self.label_vector = label
        #counts = torch.histc(label, bins=10)
        #print(counts)
        if self.label is None or self.label.shape[0] != label.shape[0]:
            self.label = torch.zeros(label.shape[0], self.num_labels,  dtype=torch.float32, requires_grad=False)
            self.label = self.label.to(device=label.device)

        self.label.zero_()
        self.label.scatter_(1, label.view(-1, 1), 1)

        self.label -= self.label.mean(0)
        std = self.label.std(0)
        std[std==0.0] = 1.0
        self.label /= std

    def generate_heatmap(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        print('Generating heatmaps...')
        for name, corr in self.name2corr.items():
            data = corr.mean([1,2]).cpu().numpy()

            sb.heatmap(data, vmin=-0.8, vmax=0.8)
            plt.savefig(os.path.join(path, '{0}_{1}.png'.format(name, self.iter)))
            plt.clf()
        self.iter += 1

        for name, corr in self.name2corr.items():
            imgpath = os.path.join(path, '{0}_%d.png'.format(name))
            movie_path = os.path.join(path, '{0}.mp4'.format(name))
            os.system("ffmpeg -r 3 -i {0} -vcodec mpeg4 -y {1} -v 0".format(imgpath, movie_path))

    def compute_layer_accuracy(self):
        print('TRAIN:')
        print('='*80)
        for name, acc in self.name2train_accs.items():
            print(name, np.median(acc), np.mean(acc), acc[-5:])
        print('VALID:')
        print('='*80)
        for name, acc in self.name2val_accs.items():
            print(name, np.median(acc), np.mean(acc), acc[-5:])

        print('ENSEMBLE', np.median(self.val_accs_ensemble), np.mean(self.val_accs_ensemble), self.val_accs_ensemble[-5:])


        self.name2train_accs = {}
        self.name2val_accs = {}
        self.val_accs_ensemble = []

    def generate_clusters(self):
        for name, corr in self.name2corr.items():
            print('Structuring layer: {0}'.format(name))
            data = corr.mean([1, 2]).cpu().numpy()
            layer = self.name2module[name]
            w = layer.weight.data
            #data = torch.cat([w.view(w.shape[0], -1), corr.view(corr.shape[0], -1)], dim=1).cpu().numpy()
            #data = w.view(w.shape[0], -1).cpu().numpy()
            #data = corr.view(corr.shape[0], -1).cpu().numpy()
            print('Scaling...')
            #data = StandardScaler().fit_transform(data)
            data = MinMaxScaler(feature_range=(-0.8, 0.8)).fit_transform(data)
            print('Fitting k-Means...')
            clf = KMeans(10, verbose=0, n_jobs=-1, tol=0, n_init=30, max_iter=300)
            #clf = KMeans(self.num_labels, verbose=1, n_jobs=-1, tol=1e-06, n_init=30)
            #clf = GaussianMixture(self.num_labels, verbose=1, tol=1e-06, max_iter=300, n_init=20, )
            #clf = AgglomerativeClustering(n_clusters=None, linkage='complete', affinity='l1', distance_threshold=0.01)
            #clf = AgglomerativeClustering(self.num_labels, linkage='complete', affinity='l1')
            #clf = DBSCAN()
            t0 = time.time()
            clf = clf.fit(data)
            clusters = clf.predict(data)
            #clusters = clf.labels_
            self.name2clusters[name] = torch.from_numpy(clusters).to(corr.device)
            #print(clf.cluster_centers_)
            #labels = torch.eye(self.num_labels).to(corr.device)
            #labels[labels == 0.0] = -1.0/self.num_labels
            #clusters = torch.from_numpy(clf.cluster_centers_).to(corr.device)

            # labelxlabel cluster feats -> label x feats
            #preds = torch.mm(labels, clusters.T)
            #print(preds.shape)
            #val, idx = torch.topk(preds, dim=0, k=3, largest=True) # max over label to get the clusters for each label
            #print('='*80)
            #print(idx)
            #print(val)

            #print(np.argmax(cluster2labels))

    def ensemble(self, name):
        if name == 'features.40' and len(self.val_votes) > 0:
            # -> bcv
            votes = torch.cat(self.val_votes, 2)
            votes = votes.view(votes.shape[0], -1)
            #out = self.model(votes)
            out = torch.einsum('bf, cf->bc', votes, F.sigmoid(self.w.weight))
            #out = torch.einsum('bcv, hv->bc', votes, F.sigmoid(self.w.weight))
            #votes = self.w(votes)

            vals, idx = torch.topk(out, k=1, dim=1)
            #vals, idx = torch.mode(idx.squeeze())
            #idx = self.clf.predict(votes)
            #idx = torch.from_numpy(idx).to(self.label_vector.device)

            self.val_accs_ensemble.append((idx.squeeze()== self.label_vector).sum().item()/self.label_vector.numel())
            self.val_votes = []

        if name == 'features.40' and len(self.train_votes) > 0:
            votes = torch.cat(self.train_votes, 2)
            #vals, idx = torch.topk(votes, k=1, dim=1)
            #modes, idx = torch.mode(idx.squeeze())

            #vals, m = torch.topk(votes, k=1, dim=1)
            #vals, modes = torch.topk(votes.mean(2), k=1, dim=1)

            votes = votes.view(votes.shape[0], -1)
            self.opt.zero_grad()
            out = torch.einsum('bf, cf->bc', votes, F.sigmoid(self.w.weight))
            #out = torch.einsum('bcv, hv->bc', votes, F.sigmoid(self.w.weight))
            #out = self.model(votes)
            #print(modes[:4])
            #print(self.label_vector[:4])
            loss = self.model.loss(out, self.label_vector)
            loss.backward()
            self.opt.step()

            #votes = torch.einsum('bf, cf->bc', votes, F.sigmoid(self.w.weight))
            #print(votes[0], self.label_vector[0])
            #votes = self.w(votes).squeeze()
            #print(votes[0], 'vote0')
            #print(self.label_vector[0], 'lbl0')
            #vals, idx = torch.topk(out, k=1, dim=1)
            #print((idx.squeeze()== self.label_vector).sum().item()/self.label_vector.numel())
            #loss = F.nll_loss(F.log_softmax(votes, dim=1), self.label_vector)
            #loss.backward()
            #print('pre', self.w.weight.data)
            #self.opt.step()
            #print('post', self.w.weight.data)
            #self.opt.zero_grad()
            #self.clf.fit(votes, self.label_vector.cpu().numpy())

            self.train_votes = []
            self.lbls = []

    def predict(self, name, activation, topk=1, is_train=True):
        corr = self.name2corr[name].clone()
        #corr[torch.abs(corr) < 0.2] = 0.0
        #preds = torch.mm(activation, corr)
        #print(corr.shape, activation.shape)
        #cov = torch.einsum('fhc,fhc->f',corr, corr)
        #preds_cov = torch.einsum('bfh,f->bf',activation, cov)
        #preds = torch.einsum('bf,fhc->bc',preds_cov, corr)
        #preds = torch.einsum('bfh,fhc->bc',activation, corr)
        preds = torch.einsum('bfhw,fhwc->bc',activation, corr)
        #cov = torch.mm(corr, corr.T)
        #preds = torch.mm(torch.mm(activation, cov), corr)
        if is_train:
            self.train_votes.append(preds.data.unsqueeze(-1))
        else:
            self.val_votes.append(preds.data.unsqueeze(-1))

        self.ensemble(name)

        val, ids = torch.topk(preds, k=topk, dim=1, largest=True)
        correct = 0
        for i in range(topk):
            correct += (self.label_vector == ids[:, i]).sum().item()

        acc = correct/self.label_vector.numel()

        if is_train:
            if name not in self.name2train_accs: self.name2train_accs[name] = []
            self.name2train_accs[name].append(acc)
        else:
            if name not in self.name2val_accs: self.name2val_accs[name] = []
            self.name2val_accs[name].append(acc)

        return preds
        # cutoff based on class probability distribution
        #return F.normalize(preds)

    def predict_with_features(self, name, activation):
        full_acc = self.name2train_accs[name][-1]
        corr = self.name2corr[name].clone()
        #corr[torch.abs(corr) < 0.2] = 0.0
        labels = torch.eye(self.num_labels).to(corr.device)
        labels[labels == 0.0] = -1.0/self.num_labels
        #labels[labels == 0.0] = -1.0
        #preds = torch.mm(activation, corr)
        #feats = torch.mm(preds, corr.T)
        feats = torch.mm(labels, torch.mm(corr.T, torch.mm(corr, corr.T)))

        feats_sorted = torch.argsort(feats, dim=1, descending=True)
        partial_corr = torch.zeros_like(corr)

        for k in range(activation.shape[1]):
            partial_corr.zero_()
            idx = feats_sorted[:, :k].reshape(-1)
            partial_corr[idx] = corr[idx]

            preds = torch.mm(activation, partial_corr)

            val, ids = torch.topk(preds, k=1, dim=1, largest=True)
            correct = 0
            correct += (self.label_vector == ids[:, 0]).sum().item()

            acc = correct/self.label_vector.numel()
            if k < 16 or k % 16 == 0:
                print('Acc for {1} features: {0}. Acc ratio: {2}'.format(acc, k, acc/full_acc))

    def make_generalists_clusters(self, num_cluster=10):
        # pick max 1/importance rank * 1/uncorrelated rank *1/num usage 
        for name, corr in self.name2corr.items():
            k = corr.shape[0] // num_cluster
            print('Finding generalists clustser for layer: {0}'.format(name))
            # [feat, h, w, class]

            r = corr.clone()
            r -= r.mean(0)
            std = r.std(0)
            std[std==0] = 1.0
            r /= std

            featR = torch.abs(torch.einsum('fhwc,vhwc->fv', r, r))
            print(featR[0])
            #featR *= (torch.eye(r.shape[0]) == 0).float()

            class_contributions = torch.zeros(self.num_labels).to(corr.device)
            unused = torch.ones(r.shape[0], 1, 1, 1).to(corr.device)

            cluster = torch.ones(corr.shape[0]).to(corr.device)*-1.0
            cluster_idx = []
            target_class = 0
            for i in range(num_cluster):
                k = (corr.shape[0] - len(cluster_idx)) if i == (num_cluster-1) else k
                for j in range(k):
                    #print(class_contributions)
                    if target_class == self.num_labels: target_class = 0
                    #print(target_class, class_contributions)
                    #if len(cluster_idx) > 0:
                    #    cross_corr = featR[torch.tensor(cluster_idx)].sum(0)
                    #else:
                    #    cross_corr = torch.ones(corr.shape[0]).to(corr.device)

                    score = (corr*unused)[:, :, :, target_class].mean([1, 2])
                    feat_idx = torch.argmax(torch.abs(score))
                    unused[feat_idx] = 0
                    #class_contributions += torch.abs(corr[feat_idx].mean([0,1]))
                    cluster_idx.append(feat_idx)
                    cluster[feat_idx] = target_class
                    target_class += 1

            self.name2clusters[name] = cluster


    def rearrange(self, name, clusters, module, corr):
        print('Rearranging neurons...')
        # do not rearrange last layer since the fully connected layer gets confused by the rearrangement
        self.name2prev_clusters[name] = clusters
        if self.prev_idx is not None:
            if module.weight.data.shape[1] == self.prev_idx.shape[0]:
                module.weight.data = module.weight.data[:, self.prev_idx]
        #if len(self.name2clusters) == 0:
        #    print('end', name)
        #    self.prev_idx = None
        #    return
        n = clusters.numel()
        idx_map = []
        val = torch.unique(clusters)
        for i in range(val.numel()):
        #for i in range(self.num_labels):
            lbl_idx = torch.where(clusters==i)[0].view(-1, 1)
            idx_map.append(lbl_idx)

        idx = torch.cat(idx_map, 0).view(-1)

        self.name2corr[name] = corr[idx]
        module.weight.data = module.weight.data[idx]
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data[idx]
        self.prev_idx = idx
        self.name2prev_clusters[name] = idx

    def forward_other(self, name, module, inputs):

        if name in self.name2clusters:
            self.rearrange(name, self.name2clusters.pop(name), module, self.name2corr[name])

        if self.prev_idx is not None:
            print(name, 'fixup')
            if isinstance(module, nn.BatchNorm2d):
                if len(self.name2clusters) == 0: print(self.prev_idx.shape, name, module.weight.shape, 'norm')
                if module.weight.shape[0] == self.prev_idx.shape[0]:
                    module.weight.data = module.weight.data[self.prev_idx]
                    module.bias.data = module.bias.data[self.prev_idx]
                    module.running_mean.data = module.running_mean.data[self.prev_idx]
                    module.running_var.data = module.running_var.data[self.prev_idx]
                    #self.prev_idx = None
                    #module.reset_running_stats()
            elif isinstance(module, nn.Linear):
                #if len(self.name2clusters) == 0: print(self.prev_idx.shape, name, module.weight.shape, 'linear')
                if len(self.name2clusters) == 0:
                    print(self.prev_idx.shape, module.weight.data.shape, inputs[0].shape, inputs[0].stride())
                    w = module.weight.data
                    s1 = w.shape[1]
                    module.weight.data = (module.weight.data.view(w.shape[0], self.prev_idx.shape[0], -1)[:, self.prev_idx]).view(w.shape)
                    #module.weight.data = module.weight.data[:, self.prev_idx]
                    #module.bias.data = module.bias.data[self.prev_idx]
                    self.prev_idx = None
            elif isinstance(module, nn.BatchNorm1d):
                #module.reset_running_stats()
                pass


    def forward_conv(self, name, module, inputs, activation):
        #print((inputs[0] == 0.0).sum().item(), (inputs[0] != 0.0).sum().item(), name)

        x = activation.data.clone()
        #print(x.shape)
        #target_elements = 2048
        #num_ele = np.prod(x.shape[1:])
        #reduction = int(num_ele/target_elements)
        #print(reduction)
        #x = x.view(x.shape[0], x.shape[1], -1)
        #print(x.shape)
        #x = torch.chunk(x,reduction, dim=2)
        #print(x.shape)
        #x = x.sum(-1)
        #print(x.shape)
        #print('=========')

        #x -= x.mean([0, 1])
        #std = x.std([0, 1])
        #std[std==0] = 1.0
        #x /= std
        #x = x.sum([2])

        #x = x.sum([2, 3])

        x -= x.mean(0)
        std = x.std(0)
        std[std==0] = 1.0
        x /= std
        if name in self.name2corr:
            preds = self.predict(name, x, topk=1, is_train=module.training)
            #self.predict_with_features(name, x)
        if module.training:
            #corr = torch.mm(x.T, self.label)/x.shape[0]
            #corr = torch.einsum('bf,bc->fc', x, self.label)/x.shape[0]
            #corr = torch.einsum('bfh,bc->fhc', x, self.label)/x.shape[0]
            corr = torch.einsum('bfhw,bc->fhwc', x, self.label)/x.shape[0]
            if name not in self.name2corr:
                self.name2corr[name] = corr
            else:
                m = self.name2corr[name]
                self.name2corr[name] = m*self.momentum + ((1.0-self.momentum)*corr)
            corr = self.name2corr[name]
        else:
            if name in self.name2corr:
                corr = self.name2corr[name]
                #self.cluster_drop(name, activation)
                #self.all_class_correlation_pruning(corr, x, activation, 0.1)
                #self.cluster_drop(name, activation)
        #self.max_class_correlation_pruning(preds, corr, activation, 0.2, 3, np.mean(self.name2accs[name]))

    def cluster_drop(self, name, activation, num_layers=100):
        #names = list(self.name2corr.keys())
        #names = set(names[-3:])
        #if name not in names: return
        k = 2
        if name in self.name2prev_clusters:
            #print('DROPPING {1} CLUSTERS FOR LAYER {0}'.format(name, k))
            clusters = self.name2prev_clusters[name]
            for i in range(k):
                idx = clusters == self.num_labels-(1+i)
                activation[:, idx] = 0.0

    def all_class_correlation_pruning(self, corr, x, activation, fraction):
        preds = torch.mm(torch.mm(x, torch.mm(corr, corr.T)), corr)
        #preds = torch.mm(x, torch.mm(corr.T, torch.mm(corr, corr.T)))
        # cutoff based on class probability distribution
        #normed_preds = F.normalize(preds)
        #normed_corr = F.normalize(corr.clone())
        #feats = torch.mm(normed_preds, normed_corr.T)
        #feats = torch.mm(normed_preds, corr.T)
        #feats = torch.mm(torch.mm(preds, corr.T), torch.mm(corr, corr.T))
        feats = torch.mm(preds, corr.T)
        #print(feats.shape)
        n = x.shape[1]
        top_feats = math.ceil(n*fraction)
        feat_val, feat_idx = torch.topk(feats, k=top_feats, largest=False, dim=1)
        #print(feat_val[:, -1])
        mask = torch.zeros_like(feats)
        counts = torch.histc(feats, bins=100, min=-3.0, max = 3.0)
        #print('='*80)
        #print(feat_val[:, -1])
        #print(counts.long())
        #print(np.linspace(-3.0, 3.0, 100))


        mask[feats >= feat_val[:, -1].view(-1, 1)] = 1.0
        mask[feats < feat_val[:, -1].view(-1, 1)] = 0.0
        #print((mask==0.0).sum().item()/ ((mask==0.0).sum().item() + (mask==1.0).sum().item()))
        activation *= activation*mask.view(mask.shape[0], mask.shape[1], 1, 1)

    def all_class_correlation_selection(self, corr, x, activation, fraction):
        n = x.shape[1]
        k = math.ceil(n*fraction)#/self.num_labels)

        labels = torch.eye(self.num_labels).to(corr.device)
        labels[labels == 0.0] = -1.0/self.num_labels
        #labels[labels == 0.0] = -1.0
        #preds = torch.mm(activation, corr)
        #feats = torch.mm(preds, corr.T)
        feats = torch.mm(labels, torch.mm(corr.T, torch.mm(corr, corr.T)))

        feats_sorted = torch.argsort(feats, dim=1, descending=True)
        partial_corr = torch.zeros_like(corr)

        idx = feats_sorted[:, -k:]
        idx = idx.reshape(-1).unique()
        activation[:, idx] = 0.0


    def max_class_correlation_pruning(self, preds, corr, activation, fraction, topk=3, accs=None):
        top_val, top_preds = torch.topk(preds, k=topk, dim=1, largest=True)

        #fraction = 1.0-accs
        labels = torch.ones(top_preds.shape[0], self.num_labels,  dtype=torch.float32, requires_grad=False)*-1
        labels = labels.to(device=top_preds.device)

        labels.scatter_(1, top_preds, 1)

        #print(labels[:3])
        feats = torch.mm(labels, corr.T)
        n = activation.shape[1]
        top_feats = math.ceil(n*fraction)
        feat_val, feat_idx = torch.topk(feats, k=top_feats, largest=False, dim=1)
        mask = torch.zeros_like(feats)
        mask[feats >= feat_val[:, -1].view(-1, 1)] = 1.0
        mask[feats < feat_val[:, -1].view(-1, 1)] = 0.0
        #print((mask==0.0).sum().item()/ ((mask==0.0).sum().item() + (mask==1.0).sum().item()))
        #feats = (feats > feat_val[:, -1].view(-1, 1)).float().view(feats.shape[0], feats.shape[1], 1, 1)
        #print((feats==0.0).sum().item(), (feats!=0.0).sum().item(), 'feats')
        #activation.data = activation.data*feats.view(feats.shape[0], feats.shape[1], 1, 1)
        activation*= feats.view(feats.shape[0], feats.shape[1], 1, 1)
        #activation= activation*feats





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
