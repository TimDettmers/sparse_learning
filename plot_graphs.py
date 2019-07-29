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
plt.errorbar(mnist['Sparsity'], mnist['Full Dense'], yerr=mnist['error1']*percentile95, fmt='.k', capsize=5)
plt.errorbar(mnist['Sparsity'], mnist['Dynamic Sparse'], yerr=mnist['error2']*percentile95, fmt='.k', ecolor=blue, capsize=5)
plt.errorbar(mnist['Sparsity'], mnist['Sparse Momentum'], yerr=mnist['error3']*percentile95, fmt='.k', ecolor=orange, capsize=5)
plt.errorbar(mnist['Sparsity'], mnist['SET'], yerr=mnist['error4']*percentile95, fmt='.k', ecolor=purple, capsize=5)
plt.errorbar(mnist['Sparsity'], mnist['DEEP-R'], yerr=mnist['error5']*percentile95, fmt='.k', ecolor=yellow, capsize=5)

plt.ylim(0.975*factor, 0.990*factor)
plt.xlim(0.00*factor, 0.21*factor)
plt.xticks([1, 2, 3, 4, 5, 20])
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
#print(stats.normaltest(np.log10(dense_data+1-dense_data.min())))
print(stats.wilcoxon(sparse_data, dense_data))

