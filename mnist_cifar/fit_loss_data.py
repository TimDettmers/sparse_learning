import argparse
from sde import LossModel, append_global_sde_args

parser = argparse.ArgumentParser('Fit loss model.')
append_global_sde_args(parser)

args = parser.parse_args()

model = LossModel(args, './loss_data', './loss_models', 'cifar-10', from_frac=0.05, to_frac=0.2)


model.fit()





