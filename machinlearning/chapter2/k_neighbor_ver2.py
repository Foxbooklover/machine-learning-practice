from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import mglearn
import matplotlib.pyplot as plt


X,y = mglearn.datasets.make_forge()

'''
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
print("{:.2f}".format(clf.score(X_test,y_test)))
'''
fig, axes = plt.subplots(1, 3, figsize=(10,3))

for n_neighbors, ax in zip([1,3,9],axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)

plt.show()