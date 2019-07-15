import glob
import numpy as np

for log_name in glob.iglob('./logs/*.log'):
    losses = []
    accs = []
    with open(log_name) as f:
        for line in f:
            if not line.startswith('Test evaluation'): continue
            loss = float(line[31:37])
            acc = float(line[61:-3])/100

            losses.append(loss)
            accs.append(acc)

    acc_std = np.std(accs, ddof=1)
    acc_se = acc_std/np.sqrt(len(accs))

    loss_std = np.std(losses, ddof=1)
    loss_se = loss_std/np.sqrt(len(losses))


    print('='*85)
    print('Test set results for log: {0}'.format(log_name))
    print('\tAccuracy. Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}'.format(np.mean(accs), acc_se, len(accs)))
    print('\tLoss.     Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}'.format(np.mean(losses), loss_se, len(losses)))
    print('='*85)
