import numpy as np
np.set_printoptions(suppress=True)

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize, OneHotEncoder


lbl = np.load('300_0.1_lbl.npy')
lbl2 = np.load('300_0.2_lbl.npy')
x1 = np.load('300_0.1.npy')
x2 = np.load('300_0.2.npy')
x3 = np.load('300_0.5.npy')
x5 = np.load('lenet_0.1.npy')
x6 = np.load('lenet_0.2.npy')
x7 = np.load('lenet_0.5.npy')

x1 = np.vstack([x1, x5])
x2 = np.vstack([x2, x6])
x3 = np.vstack([x3, x7])
lbl = np.vstack([lbl.reshape(-1, 1), lbl.reshape(-1, 1)]).reshape(-1)

enc = OneHotEncoder()
onehot = enc.fit_transform(lbl.reshape(-1, 1)).todense()
print(onehot.shape)

y = np.load('300_1.0.npy')
y = np.vstack([y, y])

print(x1.shape, lbl.shape, y.shape, onehot.shape)

eps = 1e-10
#stack = np.hstack([x1, x2, x3, x1-x2, x1-x3, x2-x3, x1/(x2+eps), x2/(x3+eps), x1/(x3+eps), onehot])
stack = np.hstack([x1, x2, x3, x1-x2, x1-x3, x2-x3, x1/(x2+eps), x2/(x3+eps), x1/(x3+eps), onehot])

#stack = normalize(stack)

X, Xtest, y, ytest, lbl, lbltest = train_test_split(stack, y, lbl, test_size=0.2, random_state=1)


clf = Ridge(alpha=14.05, normalize=True)
#clf = RandomForestRegressor(n_estimators=10)
clf.fit(X, y)

preds = clf.predict(Xtest)


y1 = np.argmax(preds, 1)
y2 = np.argmax(ytest, 1)

acc1 = np.sum((y1 == lbltest))/y1.size
acc2 = np.sum((y2 == lbltest))/y2.size
print(acc1)
print(acc2)


lbl = np.load('300_0.1_lbl.npy')
lbl2 = np.load('300_0.2_lbl.npy')
x1 = np.load('300_0.1.npy')
x2 = np.load('300_0.2.npy')
x3 = np.load('300_0.5.npy')
x4 = np.load('300_1.0.npy')
x5 = np.load('lenet_0.1.npy')
x6 = np.load('lenet_0.2.npy')
x7 = np.load('lenet_0.5.npy')
x8 = np.load('lenet_1.0.npy')

y = np.vstack([x2, x3, x4])
enc = OneHotEncoder()
onehot = enc.fit_transform(lbl.reshape(-1, 1)).todense()

x1 = np.hstack([x1, np.ones((x1.shape[0], 1))*0.1, np.ones((x1.shape[0], 1))*0.2, onehot])
x2 = np.hstack([x2, np.ones((x1.shape[0], 1))*0.2, np.ones((x1.shape[0], 1))*0.5, onehot])
x3 = np.hstack([x3, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*1.0, onehot])
x7 = np.hstack([x7, np.ones((x1.shape[0], 1))*0.5, np.ones((x1.shape[0], 1))*1.0, onehot])

stack = np.vstack([x1, x2, x3])

lbl = np.tile(lbl, 3)

print(stack.shape, y.shape, lbl.shape)


X, Xtest, y, ytest, lbl, lbltest = train_test_split(stack, y, lbl, test_size=0.2, random_state=1)


clf = Ridge(alpha=14.8, normalize=True)
#clf = RandomForestRegressor(n_estimators=10)
clf.fit(X, y)

preds = clf.predict(Xtest)


y1 = np.argmax(preds, 1)
y2 = np.argmax(ytest, 1)

acc1 = np.sum((y1 == lbltest))/y1.size
acc2 = np.sum((y2 == lbltest))/y2.size
print(acc1)
print(acc2)


preds = clf.predict(x7)
y1 = np.argmax(preds, 1)
y2 = np.argmax(x8, 1)

acc1 = np.sum((y1 == lbl2))/y1.size
acc2 = np.sum((y2 == lbl2))/y2.size
print(acc1)
print(acc2)
