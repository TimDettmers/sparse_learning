from __future__ import print_function
import sys
import os
import math
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
from torch import nn

from sde import append_global_sde_args, LossHook, LossModel

from extensions import magnitude_variance_pruning, variance_redistribution
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, plot_class_feature_histograms

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

    log_path = './logs/{0}_{1}.log'.format(args.model, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))

def evaluate(args, model, device, test_loader, is_test_set=False, loss=F.nll_loss):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    hook = LossHook.get_instance()
    with torch.no_grad():
        for data, target in test_loader:
            hook.add_input_batch(data)
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += loss(output, target).item() # sum up batch loss
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
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    append_global_sde_args(parser)

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
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        else:
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads, subset_size=args.sde_subset_size)

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

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_frequency, gamma=0.1)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

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
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))
        else:
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        hook = LossHook.get_instance()
        hook.initalize('./loss_data', args, 'cifar-10', args.sde_subset_size, '/home/tim/git/sde/config/args_config.txt')
        loss_model = LossModel(args, './loss_data', './loss_models', 'cifar-10', from_frac=0.05, to_frac=0.2)
        if args.sde_train_loss_model:
            loss_model.load_or_fit()
        loss = hook.wrap_loss(F.nll_loss)
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch)
            lr_scheduler.step()

            #if args.valid_split > 0.0:
            #    hook.stage()
            #    val_acc = evaluate(args, model, device, valid_loader, loss=loss)
            #    hook.commit()

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer' : optimizer.state_dict()},
                            is_best=False, filename=args.save_model)

            print_and_log('Current learning rate: {0:.4f}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

        hook.stage()
        evaluate(args, model, device, valid_loader, is_test_set=True, loss=loss)
        hook.commit()
        pred = loss_model.predict(hook)
        print('Predicted loss: {0:.2f}'.format(pred))
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

if __name__ == '__main__':
   main()
