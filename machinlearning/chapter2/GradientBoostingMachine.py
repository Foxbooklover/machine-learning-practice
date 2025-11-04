from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import numpy as np
import mglearn
import matplotlib.pyplot as plt


'''
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
gbrt=GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print(f"{gbrt.score(X_train, y_train) : .3f}")
print(f"{gbrt.score(X_test, y_test) : .3f}")
'''

X, y = make_blobs(centers=4, random_state=8)
##linear_svm = LinearSVC().fit(X,y)

X_new = np.hstack([X, X[:, 1:]**2])
linear_svm_3d = LinearSVC().fit(X_new, y)
