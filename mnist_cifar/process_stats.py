import argparse
import numpy as np
np.set_printoptions(precision=4, suppress=True)


from os.path import join

parser = argparse.ArgumentParser('Process stats script.')
parser.add_argument('--folder', type=str, default='./', help='Path to folder which contains the stats files.')

args = parser.parse_args()

def vectorized_accuracies(accuracy, idx2name):
    num_epochs = len(accuracy['train'][idx2name[0]])
    num_layers = len(accuracy['train'])
    train_accs = np.zeros((num_layers, num_epochs))
    valid_accs = np.zeros((num_layers, num_epochs))

    for idx in range(num_layers):
        name = idx2name[idx]
        tdata = accuracy['train'][name]
        vdata = accuracy['valid'][name]
        train_accs[idx][:] = tdata
        valid_accs[idx][:] = vdata

    return train_accs, valid_accs


files = ['accuracy.npy', 'name2corrs.npy', 'name2weights.npy']

accuracy = np.load(join(args.folder, files[0]), allow_pickle=True)[()]
name2corrs = np.load(join(args.folder, files[1]), allow_pickle=True)[()]
name2weights = np.load(join(args.folder, files[2]), allow_pickle=True)[()]

idx2name = accuracy['idx2name']
name2idx = {}

for idx, name in idx2name.items():
    name2idx[name] = idx

for idx, name in list(idx2name.items()):
    if 'Shortcut' in name:
        name1 = idx2name[idx-2]
        name2 = idx2name[idx-1]
        name2idx[name] = idx-2
        name2idx[name1] = idx-1
        name2idx[name2] = idx
        idx2name[idx-2] = name
        idx2name[idx-1] = name1
        idx2name[idx] = name2



train, val = vectorized_accuracies(accuracy, idx2name)


diff = train[1:, :] - train[:-1, :]

diff *= diff < 0.0

for epoch in range(train.shape[1]):
    #print(epoch, diff[:, epoch].tolist())
    print(train[:, epoch])
