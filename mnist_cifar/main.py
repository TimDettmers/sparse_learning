from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay, CorrelationTracker
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, plot_class_feature_histograms

from extensions import magnitude_variance_pruning, variance_redistribution

cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 10])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])
models['wrn-40-2'] = (WideResNet, [40, 2, 10, 0.3])

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask=None, tracker=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if lr_scheduler is not None: lr_scheduler.step()
        data, target = data.to(device), target.to(device)

        if tracker is not None:
            tracker.set_label(target)

        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)


        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))
    if tracker is not None:
        if args.cluster and (epoch-1) % 1 == 0:
            print('Restructing layers....')
            #tracker.generate_clusters()
            tracker.make_generalists_clusters()
        #tracker.generate_heatmap('/home/tim/data/plots/corr')

def evaluate(args, model, device, test_loader, is_test_set=False, tracker=None):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if tracker is not None: tracker.set_label(target)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--save-model', type=str, default='./models/model.pt', help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--cluster', action='store_true', help='Clusters the neurons into class groups.')
    parser.add_argument('--wave', action='store_true', help='Trains with lr-wave.')
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    print_and_log('Running with seed: {0}'.format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        else:
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
            print_and_log(model)
            print_and_log('='*60)
            print_and_log(args.model)
            print_and_log('='*60)

            print_and_log('='*60)
            print_and_log('Prune mode: {0}'.format(args.prune))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(args.redistribution))
            print_and_log('='*60)

        # add custom prune/growth/redisribution here
        if args.prune == 'magnitude_variance':
            print('Using magnitude-variance pruning. Switching to Adam optimizer...')
            args.prune = magnitude_variance_pruning
            args.optimizer = 'adam'
        if args.redistribution == 'variance':
            print('Using variance redistribution. Switching to Adam optimizer...')
            args.redistribution = variance_redistribution
            args.optimizer = 'adam'

        optimizer = None
        if args.optimizer == 'sgd':
            grps = []

            for i, p in enumerate(model.parameters()):
                grps.append({'params' : [p], 'lr' : args.lr})

            if args.wave:
                optimizer = optim.SGD(grps,lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)
            else:
                optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
            #optimizer = optim.SGD(grps,lr=0.0,momentum=0.0,weight_decay=0.0, nesterov=False)
            print('LR', args.lr)
            #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.wave:
            lr_scheduler = None
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_frequency, gamma=0.1)

        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader, tracker=tracker)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        tracker = CorrelationTracker(num_labels=10, momentum=0.9)
        tracker.build_graph(model)
        tracker.wrap_model(model)

        mask = None
        if not args.dense:
            if args.decay_schedule == 'cosine':
                decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
            elif args.decay_schedule == 'linear':
                decay = LinearDecay(args.prune_rate, len(train_loader)*(args.epochs))
            mask = Masking(optimizer, decay, prune_rate=args.prune_rate, prune_mode=args.prune, growth_mode=args.growth, redistribution_mode=args.redistribution,
                           verbose=args.verbose, fp16=args.fp16)
            mask.add_module(model, density=args.density)

        switches = [1, 4, 7, 10, 15, 20, 25, 30, 35]
        idx = [0, 1]
        lrs = np.array([0.00001, 0.03, 0.1])
        window_start = 2
        window_end = 2
        jump = 2
        layers = len(optimizer.param_groups)
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            if args.wave:
                for i, grp in enumerate(optimizer.param_groups):
                    if i in idx:
                        grp['lr'] = lrs[2]
                        print(i, grp['lr'])
                    elif i < idx[0] and i >= idx[0] - window_start:
                        grp['lr'] = lrs[1]
                        print(i, grp['lr'])
                    elif i > idx[1] and i <= idx[1] + window_end:
                        grp['lr'] = lrs[1]
                        print(i, grp['lr'])
                    else:
                        grp['lr'] = lrs[0]


                if (epoch + 1) % 3 == 0:
                    idx[0] += jump
                    idx[1] += jump
                    print(len(tracker.layerid2val))
                    if idx[1] >= layers:
                        idx[0] = 0
                        idx[1] = 1

            #optimizer.param_groups[0] = 0.1
            #optimizer.param_groups[-1] = 0.1
            lrs *= 0.90
            for i, (name, param) in enumerate(model.named_parameters()):
                pass
                #name = name.replace('.weight', '')
                #if name in tracker.name2layerid:
                #    if tracker.name2layerid[name] == tracker.stable_layer_idx:
                #print(i, tracker.stable_layer_idx)

                #if i >= tracker.stable_layer_idx*2:# and i < (tracker.stable_layer_idx*2) + 2:
                #    param.requires_grad = True
                #    print(name, 'true')
                #else:
                #    param.requires_grad = False

                #idx = ((epoch + 1) // 5)*2

                #if i >= idx:
                #    param.requires_grad = True
                #    print(name, 'true')
                #else:
                #    param.requires_grad = False

                #if i >= l-2:
                #    param.requires_grad = True
                #    print(name, 'true')





            train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask, tracker)

            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader, tracker=tracker)

            if tracker is not None:
                if epoch > 0:
                    #tracker.generate_heatmap('/home/tim/data/plots/corr'.format(args.model))
                    tracker.compute_layer_accuracy()
                    #tracker.network_class_correlation_plot('/home/tim/data/plots/network/')

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer' : optimizer.state_dict()},
                            is_best=False, filename=args.save_model)

            if not args.dense and epoch < args.epochs:
                mask.at_end_of_epoch()

            #tracker.propagate_correlations()

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

        evaluate(args, model, device, test_loader, is_test_set=True, tracker=tracker)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

if __name__ == '__main__':
   main()
