import glob
import numpy as np

for log_name in glob.iglob('./logs/*.log'):
    losses = []
    accs = []
    arg = None
    with open(log_name) as f:
        for line in f:
            if 'Namespace' in line:
                arg = line[19:-2]
            if not line.startswith('Test evaluation'): continue
            loss = float(line[31:37])
            acc = float(line[61:-3])/100

            losses.append(loss)
            accs.append(acc)
    if len(accs) == 0: continue

    acc_std = np.std(accs, ddof=1)
    acc_se = acc_std/np.sqrt(len(accs))

    loss_std = np.std(losses, ddof=1)
    loss_se = loss_std/np.sqrt(len(losses))


    print('='*85)
    print('Test set results for log: {0}'.format(log_name))
    print('Arguments:\n{0}\n'.format(arg))
    print('Accuracy. Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(accs), acc_se, len(accs),
        np.mean(accs)-(1.96*acc_se), np.mean(accs)+(1.96*acc_se)))
    print('Error.    Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(1.0-np.mean(accs), acc_se, len(accs),
        (1.0-np.mean(accs))-(1.96*acc_se), (1.0-np.mean(accs))+(1.96*acc_se)))
    print('Loss.     Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(losses), loss_se, len(losses),
        np.mean(losses)-(1.96*loss_se), np.mean(losses)+(1.96*loss_se)))
    print('='*85)
