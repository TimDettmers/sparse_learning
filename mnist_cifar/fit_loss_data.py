import argparse
import torch
import numpy as np

from sde import LossModel, append_global_sde_args

parser = argparse.ArgumentParser('Fit loss model.')
parser.add_argument('--subset-from', type=float, default=0.1, help='The subset size for the input data')
append_global_sde_args(parser)

args = parser.parse_args()

print(args)

torch.random.manual_seed(args.sde_seed)
np.random.seed(args.sde_seed)
model = LossModel(args, './loss_data', './loss_models', args.sde_name, from_frac=args.subset_from, to_frac=1.0, ip='100.96.161.67')


model.fit(evaluate=True)





