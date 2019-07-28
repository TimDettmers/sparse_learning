# Dynamic parameter reallocation in deep CNNs

The code implements the experiments in the ICML 2019 submission: Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization


## Instructions
This code implements the dynamic parameterization scheme in the ICML 2019 submission: Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization. It also implements previous  dynamic parameterization schemes such as the DeepR algorithm by [Bellec at al. 2018](https://arxiv.org/abs/1711.05136) and the SET algorithm by [Mocanu et  al. 2018](https://www.nature.com/articles/s41467-018-04316-3) as well as static parameterizations based on tied parameters similar to [the HashedNet paper](https://arxiv.org/abs/1504.04788). It also implements iterative pruning where it can take a dense model and prune it down to the required sparsity. 

The main python executable is `main.py`. Results are saved under a `./runs/` directory created at the invocation directory. An invocation of `main.py` will save various accuracy metrics as well as the model parameters in the file `./runs/{model name}_{job idx}`. Accuracy figures as well as several diagnostics are also printed out. 

### General usage
```shell
main.py [-h] [--epochs EPOCHS] [--start-epoch START_EPOCH]
               [--model {mnist_mlp,cifar10_WideResNet,imagenet_resnet50}]
               [-b BATCH_SIZE] [--lr LR] [--momentum MOMENTUM]
               [--nesterov NESTEROV] [--weight-decay WEIGHT_DECAY]
               [--L1-loss-coeff L1_LOSS_COEFF] [--print-freq PRINT_FREQ]
               [--layers LAYERS]
               [--start-pruning-after-epoch START_PRUNING_AFTER_EPOCH]
               [--prune-epoch-frequency PRUNE_EPOCH_FREQUENCY]
               [--prune-target-sparsity-fc PRUNE_TARGET_SPARSITY_FC]
               [--prune-target-sparsity-conv PRUNE_TARGET_SPARSITY_CONV]
               [--prune-iterations PRUNE_ITERATIONS]
               [--post-prune-epochs POST_PRUNE_EPOCHS]
               [--n-prune-params N_PRUNE_PARAMS] [--threshold-prune] [--prune]
               [--validate-set] [--rewire-scaling] [--tied]
               [--rescale-tied-gradient] [--rewire] [--no-validate-train]
               [--DeepR] [--DeepR_eta DEEPR_ETA]
               [--stop-rewire-epoch STOP_REWIRE_EPOCH] [--no-batch-norm]
               [--rewire-fraction REWIRE_FRACTION]
               [--sub-kernel-granularity SUB_KERNEL_GRANULARITY]
               [--cubic-prune-schedule] [--sparse-resnet-downsample]
               [--conv-group-lasso] [--big-new-weights]
               [--widen-factor WIDEN_FACTOR]
               [--initial-sparsity-conv INITIAL_SPARSITY_CONV]
               [--initial-sparsity-fc INITIAL_SPARSITY_FC] [--job-idx JOB_IDX]
               [--no-augment] [--data DIR] [-j N]
               [--copy-mask-from COPY_MASK_FROM] [--resume RESUME]
               [--schedule-file SCHEDULE_FILE] [--name NAME]

```
Optional arguments:
```
-h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  --model {mnist_mlp,cifar10_WideResNet,imagenet_resnet50}
                        network name (default: mnist_mlp)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 100)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum MOMENTUM   momentum
  --nesterov NESTEROV   nesterov momentum
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay (default: 1e-4)
  --L1-loss-coeff L1_LOSS_COEFF
                        Lasso coefficient (default: 0.0)
  --print-freq PRINT_FREQ, -p PRINT_FREQ
                        print frequency (default: 10)
  --layers LAYERS       total number of layers for wide resnet (default: 28)
  --start-pruning-after-epoch START_PRUNING_AFTER_EPOCH
                        Epoch after which to start pruning (default: 20)
  --prune-epoch-frequency PRUNE_EPOCH_FREQUENCY
                        Intervals between prunes (default: 2)
  --prune-target-sparsity-fc PRUNE_TARGET_SPARSITY_FC
                        Target sparsity when pruning fully connected layers
                        (default: 0.98)
  --prune-target-sparsity-conv PRUNE_TARGET_SPARSITY_CONV
                        Target sparsity when pruning conv layers (default:
                        0.5)
  --prune-iterations PRUNE_ITERATIONS
                        Number of prunes. Set to 1 for single prune, larger
                        than 1 for gradual pruning (default: 1)
  --post-prune-epochs POST_PRUNE_EPOCHS
                        Epochs to train after pruning is done (default: 10)
  --n-prune-params N_PRUNE_PARAMS
                        Number of parameters to re-allocate per re-allocation
                        iteration (default: 600)
  --threshold-prune     Prune based on a global threshold and not a fraction
                        (default: False)
  --prune               whether to use pruning or not (default: False)
  --validate-set        whether to use a validation set to select epoch with
                        best accuracy or not (default: False)
  --rewire-scaling      Move weights between layers during parameter re-
                        allocation (default: False)
  --tied                whether to use tied weights instead of sparse ones
                        (default: False)
  --rescale-tied-gradient
                        whether to divide the gradient of tied weights by the
                        number of their repetitions (default: False)
  --rewire              whether to run parameter re-allocation (default:
                        False)
  --no-validate-train   whether to run validation on training set (default:
                        False)
  --DeepR               Train using deepR. prune and re-allocated weights that
                        cross zero every iteration (default: False)
  --DeepR_eta DEEPR_ETA
                        eta coefficient for DeepR (default: 0.1)
  --stop-rewire-epoch STOP_REWIRE_EPOCH
                        Epoch after which to stop rewiring (default: 1000)
  --no-batch-norm       no batch normalization in the mnist_mlp
                        network(default: False)
  --rewire-fraction REWIRE_FRACTION
                        Fraction of weight to rewire (default: 0.1)
  --sub-kernel-granularity SUB_KERNEL_GRANULARITY
                        prune granularity (default: 2)
  --cubic-prune-schedule
                        Use sparsity schedule following a cubic function as in
                        Zhu et al. 2018 (instead of an exponential function).
                        (default: False)
  --sparse-resnet-downsample
                        Use sub-kernel granularity while rewiring(default:
                        False)
  --conv-group-lasso    Use group lasso to penalize an entire kernel
                        patch(default: False)
  --big-new-weights     Use weights initialized from the initial distribution
                        for the new connections instead of zeros(default:
                        False)
  --widen-factor WIDEN_FACTOR
                        widen factor for wide resnet (default: 10)
  --initial-sparsity-conv INITIAL_SPARSITY_CONV
                        Initial sparsity of conv layers(default: 0.5)
  --initial-sparsity-fc INITIAL_SPARSITY_FC
                        Initial sparsity for fully connected layers(default:
                        0.98)
  --job-idx JOB_IDX     job index provided by the job manager
  --no-augment          whether to use standard data augmentation (default:
                        use data augmentation)
  --data DIR            path to imagenet dataset
  -j N, --workers N     number of data loading workers (default: 8)
  --copy-mask-from COPY_MASK_FROM
                        checkpoint from which to copy mask data(default: none)
  --resume RESUME       path to latest checkpoint (default: none)
  --schedule-file SCHEDULE_FILE
                        yaml file containing learning rate schedule and rewire
                        period schedule
  --name NAME           name of experiment

```

### Specific experiments

The two yaml files : `wrnet_experiments.yaml` and `resnet_experiments.yaml` contain YAML lists of all the invocations of the python executable needed to run the imagenet and the CIFAR10 experiments in the paper's main text and supplementary materials. 

### Important notes

- Code development and all experiments were done with Python 3.6 and pytorch 0.4.1. 
- All experiments were conducted on NVidia TitanXP GPUs.
- Imagenet experiments require multi-GPU data parallelism, which is done by default using all available GPUs specified by environment variable `CUDA_VISIBLE_DEVICES`.