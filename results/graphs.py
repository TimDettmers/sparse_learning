import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mnist_path = './MNIST_sparse_summary.csv'

factor=100

mnist = pd.read_csv(mnist_path)
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
plt.xlim(0.00*factor, 0.11*factor)
plt.xticks([1, 2, 3, 4, 5, 10])
plt.ylabel("Test Accuracy")
plt.xlabel('Weights (%)')
plt.title("LeNet 300-100 on MNIST")

plt.show()
plt.clf()



data = pd.read_csv('./WRN-28-2_results_summary.csv')
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

plt.show()
