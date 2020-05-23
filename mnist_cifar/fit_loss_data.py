import argparse
from sde import LossModel, append_global_sde_args

parser = argparse.ArgumentParser('Fit loss model.')
append_global_sde_args(parser)

args = parser.parse_args()

model = LossModel(args, './loss_data', './loss_models', args.sde_name, from_frac=0.2, to_frac=1.0, ip='100.96.161.67')


model.fit()





