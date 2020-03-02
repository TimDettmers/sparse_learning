import glob
import numpy as np
import argparse
np.set_printoptions(suppress=True)

from os.path import join

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize, OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, PassiveAggressiveRegressor, LinearRegression, ElasticNet, Lasso

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='Use model on the test case.')

args = parser.parse_args()


models = ['alexnet-s', 'vgg-d', 'wrn-16-8']
arch2idx = {}
arch2idx['deep'] = 0
arch2idx['shallow'] = 1
arch2idx['residual'] = 2
arch2idx['mnist'] = 3
arch2idx['cifar'] = 4
arch2idx['conv'] = 5
arch2idx['fc'] = 6

def load_model_data(data_dict, model, folder_path, archs=None):
    path = join(folder_path, "{0}*.npy".format(model))
    path_lbl = join(folder_path, "{0}*lbl.npy".format(model))
    for f in glob.glob(path_lbl):
        if model not in data_dict: data_dict[model] = {}
        lbl = np.load(f)
        data_dict[model]['lbl'] = lbl

    for f in glob.glob(path):
        if 'lbl' in f: continue
        data = np.load(f)
        fraction = f.replace(folder_path, '').replace(model+'_','').replace('.npy','')
        fraction = float(fraction)
        data_dict[model][fraction] = data
        if archs is not None:
            arch = []
            for a in archs:
                if a in arch2idx:
                    arch.append(arch2idx[a])
            data_dict[model]['arch'] = np.array(arch)

def create_dataset(data, data_format=0, scaler=None, restrict_label=[0.5, 1.0], restrict_inputs=[0.1, 0.2, 0.5]):

    fractions = set()
    for model, fdict in data.items():
        for fraction in fdict:
            if isinstance(fraction, float):
                fractions.add(fraction)

    # data format 0
    # [feat_0.1, feat_0.2, relativ diff(0.1, 0.2), 0.1, 0.2, 0.5(prediction) -> feat_0.5]
    # normalize feat_0.1, feat_0.2 and feat_0.5 with the same constants, but do not normalize the relative difference and the other values
    fractions = list(fractions)
    fractions.sort()

    if scaler is None:
        x = []
        for model, fdict in data.items():
            for frac in fractions:
                x.append(data[model][frac])
        x = np.vstack(x)
        scaler = StandardScaler()
        scaler.fit(x)

    x = []
    y = []
    lbl = []
    for model, fdict in data.items():
        for start in range(len(fractions)-1):
            for end in range(2, len(fractions)):
                if end-start != 2: continue
                for label in range(end, len(fractions)):
                    if fractions[label] not in restrict_label: continue
                    f1 = fractions[start]
                    f2 = fractions[start+1]
                    if f1 not in restrict_inputs: continue
                    if f2 not in restrict_inputs: continue
                    # start = x_i
                    # end = x_i+1
                    # lbl = x_i+1+k
                    stack = []
                    for frac in fractions[start:end]:
                        stack.append(scaler.transform(fdict[frac]))
                    n = stack[0].shape[0]
                    relative_diff = f2/f1#(f2-f1)/((f2))
                    stack.append(np.ones((n, 1), dtype=np.float32)*relative_diff)
                    stack.append(np.ones((n, 1), dtype=np.float32)*f1)
                    stack.append(np.ones((n, 1), dtype=np.float32)*f2)
                    stack.append(np.ones((n, 1), dtype=np.float32)*fractions[label])
                    if 'arch' in fdict:
                        arch_mat = np.zeros((n, len(arch2idx)), dtype=np.float32)
                        arch_mat[:, fdict['arch']] = 1.0
                        stack.append(arch_mat)
                    #encoder = OneHotEncoder()
                    #stack.append(encoder.fit_transform(fdict['lbl'].reshape(-1, 1)).todense())

                    feats = np.hstack(stack)
                    lbl.append(fdict['lbl'])
                    x.append(feats)
                    y.append(scaler.transform(fdict[fractions[label]]))

    x = np.vstack(x)
    y = np.vstack(y)
    lbl = np.hstack(lbl)

    x, xtest, y, ytest, lbl, lbltest = train_test_split(x, y, lbl, test_size=0.2, random_state=1)

    return scaler, x, xtest, y, ytest, lbl, lbltest


data = {}
#load_model_data(data, 'alexnet-s', './', ['shallow', 'cifar', 'conv'])
#load_model_data(data, 'wrn-16-8', './', ['deep', 'residual', 'cifar', 'conv'])
load_model_data(data, 'vgg-d', './', ['deep', 'conv', 'cifar'])
#load_model_data(data, '300', './', ['shallow', 'mnist', 'fc'])
#load_model_data(data, 'lenet', './', ['shallow', 'conv', 'mnist'])
scaler = None
#scaler, x, xtest, y, ytest, lbl, lbltest = create_dataset(data, restrict_label=[1.0])
scaler, x, xtest, y, ytest, lbl, lbltest = create_dataset(data, restrict_label=[1.0], restrict_inputs=[0.2, 0.5])

#clf = Ridge(alpha=0.00, normalize=False)
#clf = RandomForestRegressor(n_estimators=10)
#clf = GaussianProcessRegressor()
#clf = PassiveAggressiveRegressor()
#clf = LinearRegression()
clf = ElasticNet(alpha=1.27, fit_intercept=False, normalize=False)
clf.fit(x, y)
preds = clf.predict(xtest)


y1 = np.argmax(preds, 1)
y2 = np.argmax(ytest, 1)

acc1 = np.sum((y1 == lbltest))/y1.size
acc2 = np.sum((y2 == lbltest))/y2.size
print('Accuracy predicted: {0:.4f}'.format(acc1))
print('Accuracy real: {0:.4f}'.format(acc2))

if args.test:
    data = {}
    #load_model_data(data, 'alexnet-s', './', ['shallow', 'cifar', 'conv'])
    #load_model_data(data, 'wrn-16-8', './', ['deep', 'residual', 'cifar', 'conv'])
    #load_model_data(data, 'vgg-d', './', ['deep', 'conv', 'cifar'])
    ##load_model_data(data, '300', './', ['shallow', 'mnist', 'fc'])
    #load_model_data(data, 'lenet', './', ['shallow', 'conv', 'mnist'])
    scaler, x, xtest, y, ytest, lbl, lbltest = create_dataset(data, scaler, restrict_label=[1.0], restrict_inputs=[0.2 , 0.5])
    #scaler, x, xtest, y, ytest, lbl, lbltest = create_dataset(data, scaler, restrict_label=[1.0])

    preds = clf.predict(xtest)

    y1 = np.argmax(preds, 1)
    y2 = np.argmax(ytest, 1)

    acc1 = np.sum((y1 == lbltest))/y1.size
    acc2 = np.sum((y2 == lbltest))/y2.size
    print('Accuracy predicted: {0:.4f}'.format(acc1))
    print('Accuracy real: {0:.4f}'.format(acc2))

#clf.fit(X, y)
#for model, fdict in data.items():
#    for f, data in fdict.items():
#        print(model, f, data.shape)


#lbl = np.load('300_0.1_lbl.npy')
#lbl2 = np.load('300_0.2_lbl.npy')
#x1 = np.load('300_0.1.npy')
#x2 = np.load('300_0.2.npy')
#x3 = np.load('300_0.5.npy')
#x5 = np.load('lenet_0.1.npy')
#x6 = np.load('lenet_0.2.npy')
#x7 = np.load('lenet_0.5.npy')
#
#x1 = np.vstack([x1, x5])
#x2 = np.vstack([x2, x6])
#x3 = np.vstack([x3, x7])
#lbl = np.vstack([lbl.reshape(-1, 1), lbl.reshape(-1, 1)]).reshape(-1)
#
#enc = OneHotEncoder()
#onehot = enc.fit_transform(lbl.reshape(-1, 1)).todense()
#print(onehot.shape)
#
#y = np.load('300_1.0.npy')
#y = np.vstack([y, y])
#
#print(x1.shape, lbl.shape, y.shape, onehot.shape)
#
#eps = 1e-10
##stack = np.hstack([x1, x2, x3, x1-x2, x1-x3, x2-x3, x1/(x2+eps), x2/(x3+eps), x1/(x3+eps), onehot])
#stack = np.hstack([x1, x2, x3, x1-x2, x1-x3, x2-x3, x1/(x2+eps), x2/(x3+eps), x1/(x3+eps), onehot])
#
##stack = normalize(stack)
#
#X, Xtest, y, ytest, lbl, lbltest = train_test_split(stack, y, lbl, test_size=0.2, random_state=1)
#
#
#clf = Ridge(alpha=14.05, normalize=True)
##clf = RandomForestRegressor(n_estimators=10)
#clf.fit(X, y)
#
#preds = clf.predict(Xtest)
#
#
#y1 = np.argmax(preds, 1)
#y2 = np.argmax(ytest, 1)
#
#acc1 = np.sum((y1 == lbltest))/y1.size
#acc2 = np.sum((y2 == lbltest))/y2.size
#print(acc1)
#print(acc2)
#
#
#lbl = np.load('300_0.1_lbl.npy')
#lbl2 = np.load('300_0.2_lbl.npy')
#x1 = np.load('300_0.1.npy')
#x2 = np.load('300_0.2.npy')
#x3 = np.load('300_0.5.npy')
#x4 = np.load('300_1.0.npy')
#x5 = np.load('lenet_0.1.npy')
#x6 = np.load('lenet_0.2.npy')
#x7 = np.load('lenet_0.5.npy')
#x8 = np.load('lenet_1.0.npy')
#
##y = np.vstack([x2, x3, x4, x1, x3, x4, x1, x2, x4])
#y = np.vstack([x2, x3, x4, x4, x4])
#enc = OneHotEncoder()
#onehot = enc.fit_transform(lbl.reshape(-1, 1)).todense()
#
#x12 = np.hstack([x1, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*0.2, onehot])
#x13 = np.hstack([x1, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*0.5, onehot])
#x14 = np.hstack([x1, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*1.0, onehot])
##x21 = np.hstack([x2, np.ones((x1.shape[0], 1))*0.2, np.ones((x1.shape[0], 1))*0.1, onehot])
##x23 = np.hstack([x2, np.ones((x1.shape[0], 1))*0.2, np.ones((x1.shape[0], 1))*0.5, onehot])
#x24 = np.hstack([x2, np.ones((x1.shape[0], 1))*0.2, np.ones((x1.shape[0], 1))*1.0, onehot])
##x31 = np.hstack([x3, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*0.1, onehot])
##x32 = np.hstack([x3, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*0.2, onehot])
#x34 = np.hstack([x3, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*1.0, onehot])
##x56 = np.hstack([x5, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*0.2, onehot])
##x57 = np.hstack([x5, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*0.5, onehot])
#
##x7 = np.hstack([x7, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*1.0, onehot])
##x7 = np.hstack([x6, np.ones((x1.shape[0], 1))*0.2, np.ones((x1.shape[0], 1))*1.0, onehot])
##x58 = np.hstack([x5, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*1.0, onehot])
##x58 = np.hstack([x6, np.ones((x1.shape[0], 1))*0.2, np.ones((x1.shape[0], 1))*1.0, onehot])
#x58 = np.hstack([x7, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*1.0, onehot])
#
#print(x7.shape)
#
##stack = np.vstack([x12, x23, x14, x24, x34])
##stack = np.vstack([x12, x13, x14, x21, x23, x24, x31, x32, x34])
#stack = np.vstack([x12, x13, x14, x24, x34 ])
#
#stack = normalize(stack)
#y = normalize(y)
#
#lbl = np.tile(lbl, 5)
#
#print(stack.shape, y.shape, lbl.shape)
#
#
#X, Xtest, y, ytest, lbl, lbltest = train_test_split(stack, y, lbl, test_size=0.2, random_state=1)
#
#
#clf = Ridge(alpha=0.4)#, normalize=True)
##clf = RandomForestRegressor(n_estimators=10)
#clf.fit(X, y)
#
#preds = clf.predict(Xtest)
#
#
#y1 = np.argmax(preds, 1)
#y2 = np.argmax(ytest, 1)
#
#
#for i1, i2, p1, p2 in zip(y1, y2, preds, ytest):
#    if i1 != i2:
#        print(p1[i1])
#        print(p2[i2])
#        print('='*80)
#
#
#acc1 = np.sum((y1 == lbltest))/y1.size
#acc2 = np.sum((y2 == lbltest))/y2.size
#print(acc1)
#print(acc2)
#
#
#x58 = normalize(x58)
#preds = clf.predict(x58)
#y1 = np.argmax(preds, 1)
#y2 = np.argmax(x8, 1)
#
#acc1 = np.sum((y1 == lbl2))/y1.size
#acc2 = np.sum((y2 == lbl2))/y2.size
#print(acc1)
#print(acc2)
