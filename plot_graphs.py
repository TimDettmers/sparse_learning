import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

factor=100

mnist = pd.read_csv('./results/MNIST_sparse_summary.csv')
print(mnist['Sparsity'])
mnist['Sparsity'] *= factor
mnist['Full Dense'] *= factor
mnist['Sparse Momentum'] *= factor
mnist['Dynamic Sparse'] *= factor
mnist['SET'] *= factor
mnist['DEEP-R'] *= factor
mnist['error1'] *= factor
mnist['error2'] *= factor
mnist['error3'] *= factor
mnist['error4'] *= factor
mnist['error5'] *= factor
mnist['Sparsity'] = 100-mnist['Sparsity']

#ax = sns.lineplot(x='Sparsity', y='Full Dense',data=mnist, label='Full Dense', palette=sns.color_palette("Paired", n_colors=3))
#ax = sns.lineplot(x='Sparsity', y='Dynamic Sparse',data=mnist, label='Dynamic Sparse', palette=sns.color_palette("Paired", n_colors=3))
#ax = sns.lineplot(x='Sparsity', y='Sparse Momentum',data=mnist, label='Sparse Momentum', palette=sns.color_palette("Paired", n_colors=3))
#ax.invert_xaxis()
#ax.xaxis.set_major_locator(plt.FixedLocator(mnist['Sparsity']))

percentile95 = 1.96
# color blind colors; optimized for deuteranopia/protanopia; work less well for tritanopia
orange = np.array([230, 159, 0, 255])/255.
blue = np.array([86, 180, 233, 255])/255.
purple = np.array([73, 0, 146, 255])/255.
yellow = np.array([204, 121, 167, 255])/255.
plt.plot(mnist['Sparsity'], mnist['Full Dense'], color='black')
plt.plot(mnist['Sparsity'], mnist['Dynamic Sparse'], color=blue)
plt.plot(mnist['Sparsity'], mnist['Sparse Momentum'], color=orange)
plt.plot(mnist['Sparsity'], mnist['SET'], color=purple)
plt.plot(mnist['Sparsity'], mnist['DEEP-R'], color=yellow)
plt.legend()
plt.errorbar(mnist['Sparsity'], mnist['Full Dense'], yerr=mnist['error1']*percentile95, fmt='.k', capsize=5, elinewidth=1)
plt.errorbar(mnist['Sparsity'], mnist['Dynamic Sparse'], yerr=mnist['error2']*percentile95, fmt='.k', ecolor=blue, capsize=5, elinewidth=1)
plt.errorbar(mnist['Sparsity'], mnist['Sparse Momentum'], yerr=mnist['error3']*percentile95, fmt='.k', ecolor=orange, capsize=5, elinewidth=1)
plt.errorbar(mnist['Sparsity'], mnist['SET'], yerr=mnist['error4']*percentile95, fmt='.k', ecolor=purple, capsize=5, elinewidth=1)
plt.errorbar(mnist['Sparsity'], mnist['DEEP-R'], yerr=mnist['error5']*percentile95, fmt='.k', ecolor=yellow, capsize=5)

#plt.yscale('log')
plt.ylim(0.975*factor, 0.990*factor)
plt.xlim(0.00*factor, 0.21*factor)
plt.xticks([1, 2, 3, 4, 5, 10])
plt.ylabel("Test Accuracy")
plt.xlabel('Weights (%)')
plt.title("LeNet 300-100 on MNIST")

#plt.show()
plt.clf()



data = pd.read_csv('./results/WRN-28-2_results_summary.csv')
print(data['Sparsity'])
data['Sparsity'] *= factor
data['Full Dense'] /= factor
data['Sparse Momentum'] /= factor
data['Dynamic Sparse'] /= factor
data['SET'] /= factor
data['DEEP-R'] /= factor
data['error1'] /= factor
data['error2'] /= factor
data['error3'] /= factor
data['error4'] /= factor
data['error5'] /= factor
data['Sparsity'] = 100-data['Sparsity']

percentile95 = 1.96
plt.plot(data['Sparsity'], data['Full Dense'], color='black')
plt.plot(data['Sparsity'], data['Dynamic Sparse'], color=blue)
plt.plot(data['Sparsity'], data['Sparse Momentum'], color=orange)
plt.plot(data['Sparsity'], data['SET'], color=purple)
plt.plot(data['Sparsity'], data['DEEP-R'], color=yellow)
#plt.legend()
plt.errorbar(data['Sparsity'], data['Full Dense'], yerr=data['error1']*percentile95, fmt='.k', capsize=5)
plt.errorbar(data['Sparsity'], data['Dynamic Sparse'], yerr=data['error2']*percentile95, fmt='.k', ecolor=blue, capsize=5)
plt.errorbar(data['Sparsity'], data['Sparse Momentum'], yerr=data['error3']*percentile95, fmt='.k', ecolor=orange, capsize=5)
plt.errorbar(data['Sparsity'], data['SET'], yerr=data['error4']*percentile95, fmt='.k', ecolor=purple, capsize=5)
plt.errorbar(data['Sparsity'], data['DEEP-R'], yerr=data['error5']*percentile95, fmt='.k', ecolor=yellow, capsize=5)

plt.ylim(0.927*factor, 0.95*factor)
plt.xlim(0.08*factor, 0.52*factor)
plt.xticks([10, 20, 30, 40, 50])
plt.ylabel("Test Accuracy")
plt.xlabel('Weights (%)')
plt.title("WRN 28-2 on CIFAR-10")

#plt.show()
plt.clf()

data_vgg = pd.read_csv('./results/sensivity_momentum_vgg-d.csv')
data_alexnet = pd.read_csv('./results/sensivity_momentum_alexnet-s.csv')

data_vgg = data_vgg.iloc[1:, :]
data_alexnet = data_alexnet.iloc[1:, :]

data_vgg.iloc[0:, 1:] *= 100.0
data_alexnet.iloc[0:, 1:] *= 100.0

data_alexnet.loc[0:, 'sparse SE'] *= 1.96 # 95% confidence intervals
data_alexnet.loc[0:, 'dense SE'] *= 1.96

data_vgg.loc[0:, 'sparse SE'] *= 1.96 # 95% confidence intervals
data_vgg.loc[0:, 'dense SE'] *= 1.96

print(data_vgg)
print(data_alexnet)

plt.plot(data_vgg['momentum'], data_vgg['sparse mean'], color='black', label='VGG Sparse momentum')
plt.plot(data_vgg['momentum'], data_vgg['dense mean'], color=orange, label='VGG Dense control')
#plt.plot(data_alexnet['momentum'], data_alexnet['sparse mean'], color=purple, label='AlexNet Sparse momentum')
#plt.plot(data_alexnet['momentum'], data_alexnet['dense mean'], color=yellow, label='AlexNet Dense control')
#plt.legend()
#plt.legend(bbox_to_anchor=(0, 1), loc='center right', ncol=1)
#plt.legend(bbox_to_anchor=(1.04,1), mode='expand', loc="upper left")
#l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
#l2 = plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
#l3 = plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
#l4 = plt.legend(bbox_to_anchor=(0,-0.40,1,-0.2), loc="lower left",
                #mode="expand", borderaxespad=0, ncol=2)
plt.legend()
#l5 = plt.legend(bbox_to_anchor=(1,0), loc="lower right",
                #bbox_transform=fig.transFigure, ncol=3)
#l6 = plt.legend(bbox_to_anchor=(0.4,0.8), loc="upper right")

plt.errorbar(data_vgg['momentum'], data_vgg['sparse mean'], yerr=data_vgg['sparse SE'], fmt='.k', capsize=5)
plt.errorbar(data_vgg['momentum'], data_vgg['dense mean'], yerr=data_vgg['dense SE'], fmt='.k', ecolor=orange, capsize=5)
#plt.errorbar(data_alexnet['momentum'], data_alexnet['sparse mean'], yerr=data_alexnet['sparse SE'], fmt='.k', ecolor=purple, capsize=5)
#plt.errorbar(data_alexnet['momentum'], data_alexnet['dense mean'], yerr=data_alexnet['dense SE'], fmt='.k', ecolor=yellow, capsize=5)
#plt.fill_between(data_vgg['momentum'], data_vgg['sparse mean'] - data_vgg['sparse SE'], data_vgg['sparse mean']+data_vgg['sparse SE'])#, fmt='.k', ecolor=orange, capsize=5)
#plt.fill_between(data_vgg['momentum'], data_vgg['dense mean'] - data_vgg['dense SE'], data_vgg['dense mean']+data_vgg['dense SE'])#, fmt='.k', ecolor=orange, capsize=5)

#plt.ylim(0.927*factor, 0.95*factor)
plt.xlim(0.49, 0.99)
plt.xticks([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
plt.ylabel("Test Error")
plt.xlabel('Momentum')
plt.title("Momentum Parameter Sensivity")
#plt.subplots_adjust(bottom=-0.7)
plt.tight_layout()#rect=[0,0.0,1.0,1])

plt.show()
plt.clf()



data_alexnet.loc[0:, 'sparse mean'] -= data_alexnet.loc[2, 'sparse mean']
data_alexnet.loc[0:, 'dense mean'] -= data_alexnet.loc[2, 'dense mean']
data_vgg.loc[0:, 'sparse mean'] -= data_vgg.loc[2, 'sparse mean']
data_vgg.loc[0:, 'dense mean'] -= data_vgg.loc[2, 'dense mean']

sparse_data = []
sparse_data += data_vgg.loc[:, 'sparse mean'].tolist()
sparse_data += data_alexnet.loc[:, 'sparse mean'].tolist()

dense_data = []
dense_data += data_vgg.loc[:, 'dense mean'].tolist()
dense_data += data_alexnet.loc[:, 'dense mean'].tolist()
dense_data = np.array(dense_data)

print(stats.levene(sparse_data, dense_data))
print(stats.normaltest(sparse_data))
print(stats.normaltest(dense_data))
print(stats.normaltest(np.log10(dense_data+1-dense_data.min())))
print(stats.wilcoxon(sparse_data, dense_data))

data_vgg = pd.read_csv('./results/sensivity_prune_rate_vgg-d.csv')
data_alexnet = pd.read_csv('./results/sensivity_prune_rate_alexnet-s.csv')

data_vgg.iloc[0:, 1:] *= 100.0
data_alexnet.iloc[0:, 1:] *= 100.0

data_alexnet.loc[0:, 'cosine SE'] *= 1.96 # 95% confidence intervals
data_alexnet.loc[0:, 'linear SE'] *= 1.96

data_vgg.loc[0:, 'cosine SE'] *= 1.96 # 95% confidence intervals
data_vgg.loc[0:, 'linear SE'] *= 1.96

plt.plot(data_vgg['prune_rate'], data_vgg['cosine mean'], color='black', label='Cosine annealing')
plt.plot(data_vgg['prune_rate'], data_vgg['linear mean'], color=orange, label='Linear annealing')
plt.legend()
plt.plot(data_alexnet['prune_rate'], data_alexnet['cosine mean'], color='black')#, label='Cosine annealing')
plt.plot(data_alexnet['prune_rate'], data_alexnet['linear mean'], color=orange)#, label='Linear annealing')
plt.annotate('AlexNet-s', xy=(0.25, 13.7), xytext=(0.2, 10),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('VGG16-D', xy=(0.45, 7.0), xytext=(0.40, 10),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.errorbar(data_vgg['prune_rate'], data_vgg['cosine mean'], yerr=data_vgg['cosine SE'], fmt='.k', capsize=5)
plt.errorbar(data_vgg['prune_rate'], data_vgg['linear mean'], yerr=data_vgg['linear SE'], fmt='.k', ecolor=orange, capsize=5)
plt.errorbar(data_alexnet['prune_rate'], data_alexnet['cosine mean'], yerr=data_alexnet['cosine SE'], fmt='.k', capsize=5)
plt.errorbar(data_alexnet['prune_rate'], data_alexnet['linear mean'], yerr=data_alexnet['linear SE'], fmt='.k', ecolor=orange, capsize=5)

#plt.ylim(0.927*factor, 0.95*factor)
#plt.xlim(0.49, 0.99)
plt.xticks([0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2])
plt.ylabel("Test Error")
plt.xlabel('Prune Rate')
plt.title("Prune Rate Parameter Sensivity")
#plt.subplots_adjust(bottom=-0.7)
plt.tight_layout()#rect=[0,0.0,1.0,1])

#plt.show()
plt.clf()

d = pd.read_csv('./results/MNIST_compression_comparison_lenet300-100.csv')
print(d)

labels = set(d.loc[:, 'name'].tolist())

fig, ax = plt.subplots()

#ax.set_facecolor('white')
x, y = d['density'], d['error']
for lbl in labels:
    if lbl == 'Sparse Momentum': continue
    cond = d['name'] == lbl
    plt.plot(x[cond], y[cond], linestyle='none', marker='o', label=lbl)

cond = d['name'] == 'Sparse Momentum'
plt.plot(x[cond], y[cond], color=orange, label='Sparse Momentum')
plt.plot([0,9.0], [1.34, 1.34], label='Dense (100% Weights)', color='black')
plt.legend()
plt.errorbar(x[cond], y[cond], yerr=d['sm SE'][cond]*1.96, fmt='.k', capsize=5, ecolor=orange)
plt.errorbar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1.34]*10, yerr=[0.011*1.96]*10, fmt='.k', capsize=5, ecolor='black')

names = [\
'LeCun 1989',
'Dong 2017',
'Carreira-Perpinan 2018',
'Lee 2019',
'Ullrich 2017',
'Guo 2016',
'Han 2015',
'Lee 2019',
'Molchanov 2017',
'Gomez 2018',
'Gomez 2018']

diff_pos = [\
(-0.7, 0.001),
(-0.5, 0.0),
(0, 0),
(0, 0),
(0, 0),
(-0.1, 0.04),
(-0.7, 0),
(0, 0),
(-0.1, -0.15),
(-1.0, -0.1),
(-0.0, -0.0)]

print(len(diff_pos), len(names))

print(d)

for name, x, y, diff in zip(d.loc[:, 'author'], d.loc[:, 'density'], d.loc[:, 'error'], diff_pos):
    print(name)
    if name == 'Dettmers 2019': continue
    if name == 'Dong 2017':
        ax.annotate(name, xy=(x, y), xytext=(0.5, 1.6),
                arrowprops=dict(color='black', facecolor='black',arrowstyle="-", \
                connectionstyle="angle3", lw=1), size=10)
            #arrowprops=dict(facecolor='black', shrink=0.01))
    else:
        ax.annotate(name, (x+diff[0]-0.01, y+diff[1]), size=10)
plt.ylabel("Test Error")
plt.xlabel('Weights (%)')
plt.title("LeNet 300-100 on MNIST")
#plt.subplots_adjust(bottom=-0.7)
plt.xlim(0.8, 10.5)
plt.tight_layout()#rect=[0,0.0,1.0,1])

#plt.show()
plt.clf()


d = pd.read_csv('./results/MNIST_compression_comparison_lenet5.csv')
print(d)

d = d.iloc[1:, :]

labels = set(d.loc[:, 'name'].tolist())
fig, ax = plt.subplots()
x, y = d['density'], d['error']
for lbl in labels:
    if lbl == 'Sparse Momentum': continue
    cond = d['name'] == lbl
    plt.plot(x[cond], y[cond], linestyle='none', marker='o', label=lbl)

cond = d['name'] == 'Sparse Momentum'
plt.plot(x[cond], y[cond], color=orange, label='Sparse Momentum')
plt.plot([0,10.0], [0.58, 0.58], label='Dense (100% Weights)', color='black')
plt.legend()
plt.errorbar(x[cond], y[cond], yerr=d['sm SE'][cond]*1.96, fmt='.k', capsize=5, ecolor=orange)
plt.errorbar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.58]*10, yerr=[0.01*1.96]*10, fmt='.k', capsize=5, ecolor='black')

names = [\
#'LeCun 1989',
'Dong 2017',
'Lee 2019',
'Ullrich 2017',
'Guo 2016',
'Han 2015',
'Lee 2019',
'Carreira-Perpinan 2018',
'Molchanov 2017',
'Gomez 2018',
'Gomez 2018']

diff_pos = [\
#(-0.7, 0.001),
(-0.5, 0.0),
(0, 0.03),
(0, 0),
(0.1, 0.00),
(-0.7, -0.05),
(0, 0.02),
(0.2, -0.05),
(-0.35, -0.09),
(-1.2, 0.00),
(-1.0, 0.00)]

print(len(diff_pos), len(names))

for name, x, y, diff in zip(d.loc[:, 'author'], d.loc[:, 'density'], d.loc[:, 'error'], diff_pos):
    print(name, x, y)
    if name == 'Dettmers 2019': continue

    #if name == 'Lee 2018':
    #    ax.annotate(name, xy=(x, y), xytext=(0.6, 1.2),
    #            arrowprops=dict(color='black', facecolor='black',arrowstyle="-", \
    #            connectionstyle="arc3", lw=1), size=10)
    #else:
    #    ax.annotate(name, (x+diff[0]-0.01, y+diff[1]), size=10)
    ax.annotate(name, (x+diff[0]-0.01, y+diff[1]), size=10)
plt.ylabel("Test Error")
plt.xlabel('Weights (%)')
plt.xlim(0.0, 10.5)
plt.title("LeNet-5 Caffe on MNIST")
#plt.subplots_adjust(bottom=-0.7)
plt.tight_layout()#rect=[0,0.0,1.0,1])

#plt.show()
