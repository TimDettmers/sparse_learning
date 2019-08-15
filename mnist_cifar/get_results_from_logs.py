import os
import glob
import numpy as np
import argparse
import hashlib
import copy
import shlex
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay


from os.path import join

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--all', action='store_true', help='Displays individual final results.')
parser.add_argument('--folder-path', type=str, default=None, help='The folder to evaluate if running in folder mode.')
parser.add_argument('--recursive', action='store_true', help='Apply folder-path mode to all sub-directories')
parser.add_argument('--agg-config', action='store_true', help='Aggregate same configs within folders')
parser.add_argument('--filter', type=str, help='Filters by argument.')

parser_cmd = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser_cmd.add_argument('--batch-size', type=int, default=100, metavar='N',help='input batch size for training (default: 100)')
parser_cmd.add_argument('--test-batch-size', type=int, default=100, metavar='N',help='input batch size for testing (default: 100)')
parser_cmd.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
parser_cmd.add_argument('--lr', type=float, default=0.1, metavar='LR',help='learning rate (default: 0.1)')
parser_cmd.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
parser_cmd.add_argument('--no-cuda', action='store', default=False,help='disables CUDA training')
parser_cmd.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
parser_cmd.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
parser_cmd.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
parser_cmd.add_argument('--save-model', type=str, default='./models/model.pt', help='For Saving the current Model')
parser_cmd.add_argument('--data', type=str, default='mnist')
parser_cmd.add_argument('--decay_frequency', type=int, default=25000)
parser_cmd.add_argument('--l1', type=float, default=0.0)
parser_cmd.add_argument('--fp16', action='store', help='Run in fp16 mode.')
parser_cmd.add_argument('--valid_split', type=float, default=0.1)
parser_cmd.add_argument('--resume', type=str)
parser_cmd.add_argument('--start-epoch', type=int, default=1)
parser_cmd.add_argument('--model', type=str, default='')
parser_cmd.add_argument('--l2', type=float, default=5.0e-4)
parser_cmd.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
parser_cmd.add_argument('--save-features', action='store', help='Resumes a saved model and saves its feature data to disk for plotting.')
parser_cmd.add_argument('--bench', action='store', help='Enables the benchmarking of layers and estimates sparse speedups')
parser_cmd.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
sparselearning.core.add_sparse_args(parser_cmd)



args = parser.parse_args()

if args.recursive:
    folders = [x[0] for x in os.walk(args.folder_path)]
else:
    folders = [args.folder_path if args.folder_path else './logs']

def calc_and_print_data(args, accs, losses, arg):
        acc_std = np.std(accs, ddof=1)
        acc_se = acc_std/np.sqrt(len(accs))

        loss_std = np.std(losses, ddof=1)
        loss_se = loss_std/np.sqrt(len(losses))

        print('='*85)
        print('Test set results logs in folder: {0}'.format(folder))
        print('Arguments:\n{0}\n'.format(arg))
        print('Accuracy. Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(accs), acc_se, len(accs),
            np.mean(accs)-(1.96*acc_se), np.mean(accs)+(1.96*acc_se), np.median(accs)))
        print('Error.    Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(1.0-np.mean(accs), acc_se, len(accs),
            (1.0-np.mean(accs))-(1.96*acc_se), (1.0-np.mean(accs))+(1.96*acc_se), 1.0-np.median(accs)))
        print('Loss.     Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(losses), loss_se, len(losses),
            np.mean(losses)-(1.96*loss_se), np.mean(losses)+(1.96*loss_se), np.median(losses)))
        print('='*85)

        if args.all:
            print('Individual results:')
            for loss, acc in zip(losses, accs):
                err = 1.0-acc
                print('Loss: {0:.5f}, Accuracy: {1:.5f}, Error: {2:.5f}'.format(loss, acc, err))


losses = []
accs = []
hash2accs = {}
hash2losses = {}
hash2config = {}
for folder in folders:
    losses = []
    accs = []
    for log_name in glob.iglob(join(folder, '*.log')):
        if not args.folder_path:
            losses = []
            accs = []
        arg = None
        with open(log_name) as f:
            for line in f:
                if 'Namespace' in line:
                    arg = line[10:-2]
                    if args.agg_config:
                        if len(args.filter) > 0:
                            filters = args.filter.split(' ')
                            skip = False
                            print(filters)
                            for f in filters:
                                if not f in arg: skip = True
                            if skip: continue
                        arg = ('--' + arg.replace(', ', ' --'))
                        arg = arg.replace('dense=False', 'dense')
                        arg = arg.replace('verbose=False', 'verbose')
                        arg = arg.replace('verbose=True', 'verbose')
                        arg = arg.replace('_', '-')
                        arg = arg.replace('decay-', 'decay_')
                        arg = arg.replace('valid-', 'valid_')

                        arg_str = shlex.split(arg)
                        cmd_args = parser_cmd.parse_args(arg_str)
                        args_copy = copy.deepcopy(cmd_args)
                        args_copy.iters = 1
                        args_copy.verbose = False
                        args_copy.log_interval = 1
                        args_copy.seed = 0
                        args_copy.fp16 = False
                        args_copy.max_threads = 1

                        hsval = hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()
                        if hsval not in hash2accs:
                            hash2accs[hsval] = []
                            hash2losses[hsval] = []
                            hash2config[hsval] = str(args_copy)

                if not line.startswith('Test evaluation'): continue
                try:
                    loss = float(line[31:37])
                    acc = float(line[61:-3])/100
                except:
                    print('Could not convert number: {0}'.format(line[31:37]))

                if args.agg_config:
                    hash2accs[hsval].append(acc)
                    hash2losses[hsval].append(loss)

                losses.append(loss)
                accs.append(acc)
        if len(accs) == 0: continue

        if not args.folder_path:
            calc_and_print_data(args, accs, losses, arg)

    if args.agg_config:
        for hsval in hash2accs:
            accs = hash2accs[hsval]
            losses = hash2losses[hsval]
            arg = hash2config[hsval]

            if len(accs) == 0:
                print('Test set results logs in folder {0} empty!'.format(folder))
                continue

            print(hsval)
            calc_and_print_data(args, accs, losses, arg)

    elif args.folder_path:
        if len(accs) == 0:
            print('Test set results logs in folder {0} empty!'.format(folder))
            continue

        calc_and_print_data(args, accs, losses, arg)




